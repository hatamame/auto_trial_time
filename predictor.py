"""学習済みモデルを読み込み、予測を行うモジュール。

  Stage1 (回帰アンサンブル): 試走T を予測
       ↓ (実測があれば優先、無ければ予測値を流用)
  Stage2 (回帰アンサンブル): 競走T を予測 (試走T を特徴量に含む)
  Stage2 (Ranker アンサンブル LGB+XGB): 着順スコアを算出

着順予想は Ranker スコアの z-score 平均を使用。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from db_manager import (get_bike_stats, get_bucket_stats,
                        get_player_cond_stats,
                        get_player_features, get_player_interaction_stats,
                        get_player_recent_stats, get_player_stats,
                        get_weather_defaults)
from train_model import normalize_rank

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"


class Predictor:
    def __init__(self, model_dir: Path | str = MODEL_DIR):
        d = Path(model_dir)
        self.trial_lgb = joblib.load(d / "trial_time_lgb.pkl")
        self.trial_cb = joblib.load(d / "trial_time_cb.pkl")
        self.trial_xgb = joblib.load(d / "trial_time_xgb.pkl")
        self.race_lgb = joblib.load(d / "race_time_lgb.pkl")
        self.race_cb = joblib.load(d / "race_time_cb.pkl")
        self.race_xgb = joblib.load(d / "race_time_xgb.pkl")
        self.ranker_lgb = joblib.load(d / "finish_ranker_lgb.pkl")
        self.ranker_xgb = joblib.load(d / "finish_ranker_xgb.pkl")
        self.meta = joblib.load(d / "feature_meta.pkl")
        self.cat_features: list[str] = self.meta["cat_features"]
        self.trial_features: list[str] = self.meta["trial_features"]
        self.race_features: list[str] = self.meta["race_features"]
        self.rank_features: list[str] = self.meta["rank_features"]
        self.categories: dict[str, list] = self.meta["categories"]
        self.gmt: float = self.meta.get("global_mean_trial") or 3.5
        self.gmr: float = self.meta.get("global_mean_race") or 3.55
        sk = self.meta.get("smoothing_k", {})
        self.k_player: int = sk.get("player", 30)
        self.k_bike: int = sk.get("bike", 20)
        self.k_bucket: int = sk.get("bucket", 15)

    # ------------------------------------------------------------------
    # 前処理
    # ------------------------------------------------------------------
    def _to_features(self, df: pd.DataFrame,
                     feature_cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        num_cols = [c for c in feature_cols if c not in self.cat_features]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in self.cat_features:
            df[col] = df[col].fillna("UNK").astype(str)
            df[col] = pd.Categorical(df[col],
                                     categories=self.categories.get(col))
        return df[feature_cols]

    def _cb_align(self, X: pd.DataFrame) -> pd.DataFrame:
        X2 = X.copy()
        for c in self.cat_features:
            if c in X2.columns:
                X2[c] = X2[c].astype(str).fillna("UNK")
        return X2

    @staticmethod
    def _bucket_track_temp(t) -> str:
        if t is None or (isinstance(t, float) and np.isnan(t)):
            return "UNK"
        t = float(t)
        if t < 20:
            return "cold"
        elif t < 35:
            return "cool"
        elif t < 50:
            return "warm"
        return "hot"

    @staticmethod
    def _bucket_air_temp(t) -> str:
        if t is None or (isinstance(t, float) and np.isnan(t)):
            return "UNK"
        t = float(t)
        if t < 10:
            return "cold"
        elif t < 20:
            return "cool"
        elif t < 30:
            return "warm"
        return "hot"

    @staticmethod
    def _bucket_humidity(h) -> str:
        if h is None or (isinstance(h, float) and np.isnan(h)):
            return "UNK"
        h = float(h)
        if h < 40:
            return "dry"
        elif h < 60:
            return "normal"
        elif h < 80:
            return "humid"
        return "very_humid"

    def _smooth_te(self, n: int, avg: float | None,
                   global_mean: float, k: int) -> float:
        """Bayesian smoothed target encoding。"""
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            avg = global_mean
        return (n * avg + global_mean * k) / (n + k)

    def _enrich_row(self, row: dict, defaults: dict) -> dict:
        out = dict(row)
        # rank は期別番号を捨てて S/A/B のみに正規化 (学習側と統一)
        out["rank"] = normalize_rank(out.get("rank"))
        for key in ("weather", "track_condition", "track_temp_c",
                    "air_temp_c", "humidity_pct"):
            v = out.get(key)
            if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
                out[key] = defaults.get(key)

        # --- 天候バケット (予測時にも同じ規則で計算) -----------------
        out["track_temp_bucket"] = self._bucket_track_temp(
            out.get("track_temp_c"))
        out["air_temp_bucket"] = self._bucket_air_temp(
            out.get("air_temp_c"))
        out["humidity_bucket"] = self._bucket_humidity(
            out.get("humidity_pct"))
        out["temp_weather_bucket"] = (
            f"{out['track_temp_bucket']}_"
            f"{out.get('weather') or 'UNK'}")

        # --- 選手特長 --------------------------------------------------
        pname = out.get("player_name", "")
        pf = get_player_features(pname) or {}
        out["s_power"] = pf.get("s_power")
        out["solo_power"] = pf.get("solo_power")
        out["chase_power"] = pf.get("chase_power")
        out["rain_skill"] = pf.get("rain_skill")
        out["course_pref"] = pf.get("course_pref") or "UNK"

        # --- 選手過去全期間 --------------------------------------------
        stats = get_player_stats(pname)
        n_p = stats.get("n_races") or 0
        out["player_avg_trial"] = stats.get("avg_trial_time")
        out["player_avg_race"] = stats.get("avg_race_time")
        out["player_n"] = n_p

        # --- 直近フォーム ----------------------------------------------
        rec3 = get_player_recent_stats(pname, n=3)
        rec10 = get_player_recent_stats(pname, n=10)
        out["recent3_trial_avg"] = rec3.get("recent_trial_avg")
        out["recent3_race_avg"] = rec3.get("recent_race_avg")
        out["recent10_trial_avg"] = rec10.get("recent_trial_avg")
        out["recent10_race_avg"] = rec10.get("recent_race_avg")
        out["recent10_top3_rate"] = rec10.get("recent_top3_rate")

        # --- 選手 × 会場 / 湿 -----------------------------------------
        inter = get_player_interaction_stats(pname,
                                             out.get("racetrack", ""))
        out["player_at_track_trial_avg"] = inter["player_at_track_trial_avg"]
        out["player_at_track_race_avg"] = inter["player_at_track_race_avg"]
        out["player_at_track_n"] = inter["player_at_track_n"]
        out["player_in_wet_trial_avg"] = inter["player_in_wet_trial_avg"]
        out["player_in_wet_race_avg"] = inter["player_in_wet_race_avg"]
        out["player_in_wet_n"] = inter["player_in_wet_n"]
        out["recent5_wet_trial_avg"] = inter["recent5_wet_trial_avg"]
        out["recent5_wet_race_avg"] = inter["recent5_wet_race_avg"]
        out["recent5_wet_n"] = inter["recent5_wet_n"]

        # --- is_wet フラグ (現在条件) ---------------------------------
        w_str = str(out.get("weather") or "")
        t_str = str(out.get("track_condition") or "")
        is_wet = int(
            ("雨" in w_str) or ("雪" in w_str)
            or ("湿" in t_str) or ("斑" in t_str))
        out["is_wet"] = is_wet

        # --- 雨巧拙 × 湿 補正特徴量 -----------------------------------
        rs = out.get("rain_skill")
        try:
            rs_val = float(rs) if rs is not None else 3.0
        except (TypeError, ValueError):
            rs_val = 3.0
        out["rain_skill_x_wet"] = rs_val * is_wet
        out["inv_rain_skill_x_wet"] = (6.0 - rs_val) * is_wet
        # 個人の湿時 delta (湿でなければ 0)
        if is_wet and (inter.get("player_in_wet_trial_avg") is not None
                       and stats.get("avg_trial_time") is not None):
            out["player_wet_delta_trial"] = (
                inter["player_in_wet_trial_avg"] - stats["avg_trial_time"])
        else:
            out["player_wet_delta_trial"] = 0.0
        if is_wet and (inter.get("player_in_wet_race_avg") is not None
                       and stats.get("avg_race_time") is not None):
            out["player_wet_delta_race"] = (
                inter["player_in_wet_race_avg"] - stats["avg_race_time"])
        else:
            out["player_wet_delta_race"] = 0.0

        # --- target encoding (smoothed) -------------------------------
        out["player_te_trial"] = self._smooth_te(
            n_p, stats.get("avg_trial_time"), self.gmt, self.k_player)
        out["player_te_race"] = self._smooth_te(
            n_p, stats.get("avg_race_time"), self.gmr, self.k_player)

        bike = out.get("bike_name") or "UNK"
        bs = get_bike_stats(bike)
        out["bike_te_trial"] = self._smooth_te(
            bs["n"], bs["avg_trial"], self.gmt, self.k_bike)
        out["bike_te_race"] = self._smooth_te(
            bs["n"], bs["avg_race"], self.gmr, self.k_bike)
        out["bike_n"] = bs["n"]

        gs = get_bucket_stats(out.get("racetrack", ""),
                              out["track_temp_bucket"],
                              out.get("weather"))
        out["bucket_track_te_trial"] = self._smooth_te(
            gs["n"], gs["avg_trial"], self.gmt, self.k_bucket)
        out["bucket_track_te_race"] = self._smooth_te(
            gs["n"], gs["avg_race"], self.gmr, self.k_bucket)

        # 選手 × 走路状況 の smoothed TE
        pc = get_player_cond_stats(pname, out.get("track_condition") or "UNK")
        out["player_cond_te_trial"] = self._smooth_te(
            pc["n"], pc["avg_trial"], self.gmt, self.k_bucket)
        out["player_cond_te_race"] = self._smooth_te(
            pc["n"], pc["avg_race"], self.gmr, self.k_bucket)
        return out

    def _add_race_context_local(self, df: pd.DataFrame) -> pd.DataFrame:
        """同レース内の文脈特徴量を 1 レース分の行に対して計算する。"""
        df = df.copy()
        df["handicap_m"] = pd.to_numeric(df["handicap_m"], errors="coerce")
        df["handicap_rank_in_race"] = df["handicap_m"].rank(
            method="first", ascending=False)
        df["field_size"] = len(df)
        df["handicap_diff_vs_min"] = df["handicap_m"] - df["handicap_m"].min()
        is_s = df["rank"].fillna("").astype(str).str.startswith("S")
        df["n_s_class_in_race"] = int(is_s.sum())
        df["mean_opponent_avg_trial"] = (
            pd.to_numeric(df.get("player_avg_trial"), errors="coerce").mean())
        return df

    # ------------------------------------------------------------------
    # 予測器
    # ------------------------------------------------------------------
    def _predict_trial(self, X: pd.DataFrame) -> np.ndarray:
        return ((self.trial_lgb.predict(X)
                 + self.trial_cb.predict(self._cb_align(X))
                 + self.trial_xgb.predict(X)) / 3.0)

    def _predict_race(self, X: pd.DataFrame) -> np.ndarray:
        return ((self.race_lgb.predict(X)
                 + self.race_cb.predict(self._cb_align(X))
                 + self.race_xgb.predict(X)) / 3.0)

    def _predict_rank(self, X: pd.DataFrame) -> np.ndarray:
        """レース内z-score平均で2モデルアンサンブル。"""
        s_lgb = self.ranker_lgb.predict(X)
        s_xgb = self.ranker_xgb.predict(X)
        def _z(s):
            mu, sd = s.mean(), s.std(ddof=0)
            return (s - mu) / (sd + 1e-9)
        return (_z(s_lgb) + _z(s_xgb)) / 2.0

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------
    def predict_race(self, race_info: dict, players: list[dict]) -> pd.DataFrame:
        # 天候デフォルト
        month = 1
        try:
            ymd = str(race_info.get("yyyymmdd", ""))
            if len(ymd) >= 6:
                month = int(ymd[4:6])
        except ValueError:
            pass
        defaults = get_weather_defaults(
            race_info.get("racetrack", ""), month)
        used_defaults = {}
        for key in ("weather", "track_condition", "track_temp_c",
                    "air_temp_c", "humidity_pct"):
            v = race_info.get(key)
            if v is None or v == "" or (isinstance(v, float)
                                        and np.isnan(v)):
                used_defaults[key] = defaults[key]
                race_info[key] = defaults[key]

        rows: list[dict] = []
        for p in players:
            base = {
                "racetrack": race_info.get("racetrack"),
                "pid": race_info.get("pid"),
                "race_no": race_info.get("race_no"),
                "grade": race_info.get("grade") or "普通",
                "event_type": race_info.get("event_type") or "デイ",
                "weather": race_info.get("weather"),
                "track_condition": race_info.get("track_condition"),
                "track_temp_c": race_info.get("track_temp_c"),
                "air_temp_c": race_info.get("air_temp_c"),
                "humidity_pct": race_info.get("humidity_pct"),
                "bike_no": p.get("bike_no"),
                "rank": p.get("rank") or "UNK",
                "player_lg": p.get("player_lg") or "UNK",
                "license_period": p.get("license_period"),
                "player_name": p.get("player_name", ""),
                "bike_class": p.get("bike_class") or "UNK",
                "bike_name": p.get("bike_name") or "UNK",
                "handicap_m": p.get("handicap_m"),
                "trial_time_actual": p.get("trial_time_actual"),
            }
            rows.append(self._enrich_row(base, defaults))

        df = pd.DataFrame(rows)
        df = self._add_race_context_local(df)

        # --- Stage 1: 試走T 予測 ----------------------------------------
        X_trial = self._to_features(df, self.trial_features)
        pred_trial = self._predict_trial(X_trial)

        # --- 実測試走Tがあれば優先、無ければ予測値を stacking 入力に
        # ※ pred_trial を破壊しないよう必ずコピーを取る
        eff_trial = np.array(pred_trial, dtype=float, copy=True)
        used_actual_mask = np.zeros(len(eff_trial), dtype=bool)
        for i, p in enumerate(players):
            v = p.get("trial_time_actual")
            try:
                if v is not None and not pd.isna(v):
                    eff_trial[i] = float(v)
                    used_actual_mask[i] = True
            except (TypeError, ValueError):
                pass

        df["trial_time"] = eff_trial

        # --- Stage 2: 競走T と Ranker ----------------------------------
        X_race = self._to_features(df, self.race_features)
        pred_race = self._predict_race(X_race)

        X_rank = self._to_features(df, self.rank_features)
        rank_score = self._predict_rank(X_rank)

        # 実測試走T を列として保持 (無ければ NaN)
        actual_trials: list[float | None] = []
        for p in players:
            v = p.get("trial_time_actual")
            try:
                actual_trials.append(
                    float(v) if (v is not None and not pd.isna(v))
                    else None)
            except (TypeError, ValueError):
                actual_trials.append(None)

        res = pd.DataFrame({
            "bike_no": df["bike_no"].values,
            "player_name": df["player_name"].values,
            "rank": df["rank"].values,
            "handicap_m": df["handicap_m"].values,
            # 試走T: 予測値は常に表示、実測は別列
            "pred_trial_time": np.round(pred_trial, 3),
            "actual_trial_time": [
                round(v, 3) if v is not None else None
                for v in actual_trials],
            "trial_time_used": np.round(eff_trial, 3),
            "trial_time_source": np.where(
                used_actual_mask, "実測", "予測"),
            "pred_race_time": np.round(pred_race, 4),
            "rank_score": np.round(rank_score, 4),
        })
        res["pred_finish_order"] = (
            (-res["rank_score"]).rank(method="first").astype(int))
        res["pred_finish_order_by_time"] = (
            res["pred_race_time"].rank(method="first").astype(int))
        res = res.sort_values("bike_no").reset_index(drop=True)
        res.attrs["used_defaults"] = used_defaults
        return res
