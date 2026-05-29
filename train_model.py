"""機械学習モデルの学習スクリプト。

DuckDB から学習データを取得し、以下のモデルを学習する:

  回帰アンサンブル (3モデル平均):
    - LightGBM
    - CatBoost
    - XGBoost
  ターゲット:
    1. 試走タイム (trial_time)
    2. 競走タイム (expected_race_time)   ※ 試走T を特徴量に含める (stacking)

  ランカー (着順専用、2モデル平均):
    3. LightGBM LambdaRank + XGBoost Ranker (rank:pairwise)

外部ファイル化:
  - models/trial_time_{lgb,cb,xgb}.pkl
  - models/race_time_{lgb,cb,xgb}.pkl
  - models/finish_ranker_lgb.pkl
  - models/finish_ranker_xgb.pkl
  - models/feature_meta.pkl   (前処理メタ情報)
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker, XGBRegressor

from db_manager import load_training_data

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# === モデルファイルパス ====================================================
TRIAL_LGB_PATH = MODEL_DIR / "trial_time_lgb.pkl"
TRIAL_CB_PATH = MODEL_DIR / "trial_time_cb.pkl"
TRIAL_XGB_PATH = MODEL_DIR / "trial_time_xgb.pkl"
RACE_LGB_PATH = MODEL_DIR / "race_time_lgb.pkl"
RACE_CB_PATH = MODEL_DIR / "race_time_cb.pkl"
RACE_XGB_PATH = MODEL_DIR / "race_time_xgb.pkl"
RANKER_LGB_PATH = MODEL_DIR / "finish_ranker_lgb.pkl"
RANKER_XGB_PATH = MODEL_DIR / "finish_ranker_xgb.pkl"
META_PATH = MODEL_DIR / "feature_meta.pkl"

# warm-start 互換のため旧名も維持
TRIAL_MODEL_PATH = TRIAL_LGB_PATH
RACE_MODEL_PATH = RACE_LGB_PATH

# === target encoding の smoothing 強度 ====================================
K_PLAYER = 30
K_BIKE = 20
K_BUCKET = 15

# === 特徴量定義 ============================================================
# 試走T モデル用 (trial_time は含めない)
BASE_NUM_FEATURES = [
    "pid", "handicap_m",
    "track_temp_c", "air_temp_c", "humidity_pct",
    "s_power", "solo_power", "chase_power", "rain_skill",
    # 選手過去全期間
    "player_avg_trial", "player_avg_race", "player_n",
    # 直近フォーム (rolling 3/10)
    "recent3_trial_avg", "recent3_race_avg",
    "recent10_trial_avg", "recent10_race_avg",
    "recent10_top3_rate",
    # 選手 × 会場
    "player_at_track_trial_avg", "player_at_track_race_avg",
    "player_at_track_n",
    # 選手 × 湿 (拡張: weather∈{雨,雪} OR track_condition∈{湿走路,斑走路})
    "player_in_wet_trial_avg", "player_in_wet_race_avg",
    "player_in_wet_n",
    "recent5_wet_trial_avg", "recent5_wet_race_avg",
    "recent5_wet_n",
    # 雨巧拙 × 湿条件 の補正特徴量
    "rain_skill_x_wet",
    "inv_rain_skill_x_wet",
    "player_wet_delta_trial",
    "player_wet_delta_race",
    # 同レース内コンテキスト
    "handicap_rank_in_race", "handicap_diff_vs_min",
    "field_size", "n_s_class_in_race",
    "mean_opponent_avg_trial",
    # is_wet フラグ自体も入れる (0/1)
    "is_wet",
    # Target encoding (smoothed)
    "player_te_trial", "player_te_race",
    "bike_te_trial", "bike_te_race", "bike_n",
    "bucket_track_te_trial", "bucket_track_te_race",
    "player_cond_te_trial", "player_cond_te_race",
]
CAT_FEATURES = [
    "racetrack", "grade", "event_type",
    "weather", "track_condition",
    "rank", "player_lg", "bike_class",
    "course_pref",
    # 天候バケット
    "track_temp_bucket", "air_temp_bucket",
    "humidity_bucket", "temp_weather_bucket",
]
TRIAL_FEATURES = BASE_NUM_FEATURES + CAT_FEATURES
# 競走T / Ranker モデル用 (試走T を stacking 特徴量として追加)
NUM_FEATURES_WITH_TRIAL = BASE_NUM_FEATURES + ["trial_time"]
RACE_FEATURES = NUM_FEATURES_WITH_TRIAL + CAT_FEATURES
RANK_FEATURES = RACE_FEATURES


# ===========================================================================
# 特徴量エンジニアリング
# ===========================================================================
def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


_RANK_RE = re.compile(r"^([A-Za-z]+)")


def normalize_rank(r) -> str:
    """rank から先頭の英字 (S/A/B 等) だけを抽出する。期別番号は捨てる。

    例: 'S31' -> 'S', 'A189' -> 'A', 'B121' -> 'B', '' -> 'UNK'
    """
    if r is None or (isinstance(r, float) and pd.isna(r)):
        return "UNK"
    s = str(r).strip()
    if not s:
        return "UNK"
    m = _RANK_RE.match(s)
    return m.group(1).upper() if m else "UNK"


def _add_player_history(df: pd.DataFrame) -> pd.DataFrame:
    """選手別の過去集計 (全期間 + rolling + 相互作用) を leak なしで付加。"""
    df = df.sort_values(
        ["yyyymmdd", "racetrack", "race_no", "bike_no"]
    ).reset_index(drop=True)

    grp = df.groupby("player_name", sort=False)
    # 全期間平均
    df["player_avg_trial"] = (
        grp["trial_time"].apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True))
    df["player_avg_race"] = (
        grp["expected_race_time"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True))
    df["player_n"] = grp.cumcount()

    # 直近 N 走 (rolling)
    for n in [3, 10]:
        df[f"recent{n}_trial_avg"] = (
            grp["trial_time"].apply(
                lambda s: s.shift().rolling(n, min_periods=1).mean())
            .reset_index(level=0, drop=True))
        df[f"recent{n}_race_avg"] = (
            grp["expected_race_time"].apply(
                lambda s: s.shift().rolling(n, min_periods=1).mean())
            .reset_index(level=0, drop=True))

    # 直近10走の3着内率
    if "finish_position" in df.columns:
        top3_flag = (df["finish_position"].fillna(99) <= 3).astype(int)
        df["recent10_top3_rate"] = (
            top3_flag.groupby(df["player_name"]).apply(
                lambda s: s.shift().rolling(10, min_periods=1).mean())
            .reset_index(level=0, drop=True))
    else:
        df["recent10_top3_rate"] = np.nan

    # 選手 × 会場
    pt = df.groupby(["player_name", "racetrack"], sort=False)
    df["player_at_track_trial_avg"] = (
        pt["trial_time"].apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=[0, 1], drop=True))
    df["player_at_track_race_avg"] = (
        pt["expected_race_time"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=[0, 1], drop=True))
    df["player_at_track_n"] = pt.cumcount()

    # 湿状態フラグ: 雨/雪 OR 湿走路/斑走路
    weather_str = df["weather"].fillna("").astype(str)
    track_str = df["track_condition"].fillna("").astype(str)
    is_wet = (weather_str.str.contains("雨")
              | weather_str.str.contains("雪")
              | track_str.str.contains("湿")
              | track_str.str.contains("斑")).astype(int)
    df["is_wet"] = is_wet

    # ===== 高速ベクトル化: cumsum/cumcount で expanding conditional mean =====
    # 「過去の湿レースのみ」の平均を groupby.apply 無しで計算する
    wet_t = pd.to_numeric(df["trial_time"], errors="coerce").fillna(0) * is_wet
    wet_r = pd.to_numeric(df["expected_race_time"],
                          errors="coerce").fillna(0) * is_wet

    # --- 選手 × 湿 (全期間 expanding) ---
    df["__wet_t"] = wet_t.values
    df["__wet_r"] = wet_r.values
    df["__wet_f"] = is_wet.values
    cum_pt = df.groupby("player_name", sort=False)["__wet_t"].cumsum()
    cum_pr = df.groupby("player_name", sort=False)["__wet_r"].cumsum()
    cum_pf = df.groupby("player_name", sort=False)["__wet_f"].cumsum()
    prior_pt = cum_pt - df["__wet_t"]
    prior_pr = cum_pr - df["__wet_r"]
    prior_pf = cum_pf - df["__wet_f"]
    df["player_in_wet_trial_avg"] = prior_pt / prior_pf.where(prior_pf > 0)
    df["player_in_wet_race_avg"] = prior_pr / prior_pf.where(prior_pf > 0)
    df["player_in_wet_n"] = prior_pf

    # --- 選手 × 直近5回の湿レース (rolling=5 only on wet rows) ---
    # 高速化: wet 行のみ抽出 → groupby 内 rolling → 元 df にマージ
    wet_only = df.loc[is_wet == 1, ["player_name", "trial_time",
                                    "expected_race_time"]].copy()
    if len(wet_only) > 0:
        wo_grp = wet_only.groupby("player_name", sort=False)
        wet_only["r5_t"] = wo_grp["trial_time"].apply(
            lambda s: s.shift().rolling(5, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        wet_only["r5_r"] = wo_grp["expected_race_time"].apply(
            lambda s: s.shift().rolling(5, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        wet_only["r5_n"] = wo_grp.cumcount()
        # 元 df に index ベースで戻す
        df.loc[wet_only.index, "_r5_t"] = wet_only["r5_t"]
        df.loc[wet_only.index, "_r5_r"] = wet_only["r5_r"]
        df.loc[wet_only.index, "_r5_n"] = wet_only["r5_n"]
    # 非湿行は ffill で「直前の湿レース由来の値」を引き継ぐ
    df["recent5_wet_trial_avg"] = (
        df.groupby("player_name", sort=False)["_r5_t"].ffill()
        if "_r5_t" in df.columns else np.nan)
    df["recent5_wet_race_avg"] = (
        df.groupby("player_name", sort=False)["_r5_r"].ffill()
        if "_r5_r" in df.columns else np.nan)
    df["recent5_wet_n"] = (
        df.groupby("player_name", sort=False)["_r5_n"].ffill().fillna(0)
        if "_r5_n" in df.columns else 0)
    df = df.drop(columns=[c for c in ("_r5_t", "_r5_r", "_r5_n")
                          if c in df.columns])

    df = df.drop(columns=["__wet_t", "__wet_r", "__wet_f"])

    # --- 雨巧拙 × 湿条件 の補正特徴量 ---
    # rain_skill は 1〜5 の評価 (大きいほど雨に強い)
    rain_skill = pd.to_numeric(df.get("rain_skill"),
                               errors="coerce").fillna(3.0)
    df["rain_skill_x_wet"] = rain_skill * is_wet
    df["inv_rain_skill_x_wet"] = (6.0 - rain_skill) * is_wet
    # 選手個人の湿レース時の試走T変化量 (湿の平均 - 通常の平均)
    # → 湿条件で予測値を直接補正できる
    df["player_wet_delta_trial"] = (
        df["player_in_wet_trial_avg"] - df["player_avg_trial"]
    ).where(is_wet == 1, 0.0)
    df["player_wet_delta_race"] = (
        df["player_in_wet_race_avg"] - df["player_avg_race"]
    ).where(is_wet == 1, 0.0)

    return df


def _bucket_track_temp(t) -> str:
    if pd.isna(t):
        return "UNK"
    t = float(t)
    if t < 20:
        return "cold"
    elif t < 35:
        return "cool"
    elif t < 50:
        return "warm"
    else:
        return "hot"


def _bucket_air_temp(t) -> str:
    if pd.isna(t):
        return "UNK"
    t = float(t)
    if t < 10:
        return "cold"
    elif t < 20:
        return "cool"
    elif t < 30:
        return "warm"
    else:
        return "hot"


def _bucket_humidity(h) -> str:
    if pd.isna(h):
        return "UNK"
    h = float(h)
    if h < 40:
        return "dry"
    elif h < 60:
        return "normal"
    elif h < 80:
        return "humid"
    else:
        return "very_humid"


def _add_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    """天候系のバケット列を追加 (希少組合せを補強)。"""
    df["track_temp_bucket"] = df["track_temp_c"].map(_bucket_track_temp)
    df["air_temp_bucket"] = df["air_temp_c"].map(_bucket_air_temp)
    df["humidity_bucket"] = df["humidity_pct"].map(_bucket_humidity)
    df["temp_weather_bucket"] = (
        df["track_temp_bucket"].astype(str) + "_"
        + df.get("weather", pd.Series(["UNK"] * len(df))).fillna("UNK")
            .astype(str))
    return df


def _add_target_encodings_train(df: pd.DataFrame,
                                gmt: float, gmr: float) -> pd.DataFrame:
    """Bayesian smoothed target encoding を leak-free に付加 (学習用)。

    encoded = (prior_sum + global_mean * k) / (prior_count + k)
    prior_* は当該行を含まない過去累積。
    """
    df = df.sort_values(
        ["yyyymmdd", "racetrack", "race_no", "bike_no"]
    ).reset_index(drop=True)

    # --- player --------------------------------------------------------
    grp_p = df.groupby("player_name", sort=False)
    pp_sum_t = grp_p["trial_time"].apply(
        lambda s: s.shift().expanding().sum()
    ).reset_index(level=0, drop=True)
    pp_sum_r = grp_p["expected_race_time"].apply(
        lambda s: s.shift().expanding().sum()
    ).reset_index(level=0, drop=True)
    pp_cnt = grp_p.cumcount()
    df["player_te_trial"] = (pp_sum_t + gmt * K_PLAYER) / (pp_cnt + K_PLAYER)
    df["player_te_race"] = (pp_sum_r + gmr * K_PLAYER) / (pp_cnt + K_PLAYER)

    # --- bike ----------------------------------------------------------
    bike_key = df["bike_name"].fillna("UNK").astype(str)
    grp_b = df.groupby(bike_key, sort=False)
    bb_sum_t = grp_b["trial_time"].apply(
        lambda s: s.shift().expanding().sum()
    ).reset_index(level=0, drop=True)
    bb_sum_r = grp_b["expected_race_time"].apply(
        lambda s: s.shift().expanding().sum()
    ).reset_index(level=0, drop=True)
    bb_cnt = grp_b.cumcount()
    df["bike_te_trial"] = (bb_sum_t + gmt * K_BIKE) / (bb_cnt + K_BIKE)
    df["bike_te_race"] = (bb_sum_r + gmr * K_BIKE) / (bb_cnt + K_BIKE)
    df["bike_n"] = bb_cnt

    # --- bucket × racetrack × weather ----------------------------------
    grp_bw = df.groupby(
        ["racetrack", "track_temp_bucket", "weather"], sort=False)
    bw_sum_t = grp_bw["trial_time"].apply(
        lambda s: s.shift().expanding().sum()
    ).reset_index(level=[0, 1, 2], drop=True)
    bw_sum_r = grp_bw["expected_race_time"].apply(
        lambda s: s.shift().expanding().sum()
    ).reset_index(level=[0, 1, 2], drop=True)
    bw_cnt = grp_bw.cumcount()
    df["bucket_track_te_trial"] = (
        bw_sum_t + gmt * K_BUCKET) / (bw_cnt + K_BUCKET)
    df["bucket_track_te_race"] = (
        bw_sum_r + gmr * K_BUCKET) / (bw_cnt + K_BUCKET)

    # --- player × track_condition ----------------------------------
    grp_pc = df.groupby(
        ["player_name", "track_condition"], sort=False)
    pc_sum_t = grp_pc["trial_time"].apply(
        lambda s: s.shift().expanding().sum()
    ).reset_index(level=[0, 1], drop=True)
    pc_sum_r = grp_pc["expected_race_time"].apply(
        lambda s: s.shift().expanding().sum()
    ).reset_index(level=[0, 1], drop=True)
    pc_cnt = grp_pc.cumcount()
    df["player_cond_te_trial"] = (
        pc_sum_t + gmt * K_BUCKET) / (pc_cnt + K_BUCKET)
    df["player_cond_te_race"] = (
        pc_sum_r + gmr * K_BUCKET) / (pc_cnt + K_BUCKET)
    return df


def _add_race_context(df: pd.DataFrame) -> pd.DataFrame:
    """同レース内の文脈特徴量を加える (ベクトル化)。"""
    g = df.groupby(["yyyymmdd", "pid", "race_no"], sort=False)
    df["handicap_rank_in_race"] = g["handicap_m"].rank(
        method="first", ascending=False)
    df["field_size"] = g["bike_no"].transform("count")
    df["handicap_diff_vs_min"] = (
        df["handicap_m"] - g["handicap_m"].transform("min"))
    is_s = df["rank"].fillna("").astype(str).str.startswith("S").astype(int)
    df["n_s_class_in_race"] = is_s.groupby(
        [df["yyyymmdd"], df["pid"], df["race_no"]]).transform("sum")
    df["mean_opponent_avg_trial"] = g["player_avg_trial"].transform("mean")
    return df


def prepare_features(df: pd.DataFrame,
                     global_means: dict | None = None) -> pd.DataFrame:
    """学習データから特徴量 DataFrame を作る。

    global_means が None なら現在の df から計算する (full_retrain 用)。
    指定されればそれを使う (incremental 用)。
    結果の `df.attrs` に gmt/gmr が入る。
    """
    df = df.copy()
    for col in ["pid", "race_no", "handicap_m", "track_temp_c",
                "air_temp_c", "humidity_pct",
                "trial_time", "expected_race_time", "finish_position",
                "s_power", "solo_power", "chase_power", "rain_skill"]:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])
    # bike_name は集計キーとしてのみ使用 (CAT特徴量には入れない)
    if "bike_name" not in df.columns:
        df["bike_name"] = "UNK"
    else:
        df["bike_name"] = df["bike_name"].fillna("UNK").astype(str)
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("UNK").astype(str)
        else:
            df[col] = "UNK"
    # rank は期別番号を捨てて S/A/B のみにする
    df["rank"] = df["rank"].map(normalize_rank)
    # 順序が重要: バケット → player history → race context → TE
    df = _add_bucket_features(df)
    df = _add_player_history(df)
    df = _add_race_context(df)

    if global_means is None:
        gmt = float(pd.to_numeric(df["trial_time"],
                                  errors="coerce").mean())
        gmr = float(pd.to_numeric(df["expected_race_time"],
                                  errors="coerce").mean())
    else:
        gmt = float(global_means["trial"])
        gmr = float(global_means["race"])
    df = _add_target_encodings_train(df, gmt, gmr)
    df.attrs["gmt"] = gmt
    df.attrs["gmr"] = gmr
    return df


def _to_categorical(df: pd.DataFrame, cat_cols: list[str],
                    categories: dict[str, pd.Index] | None = None
                    ) -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    cats: dict[str, pd.Index] = {}
    out = df.copy()
    for col in cat_cols:
        if categories is not None and col in categories:
            out[col] = pd.Categorical(out[col], categories=categories[col])
        else:
            out[col] = pd.Categorical(out[col])
        cats[col] = out[col].cat.categories
    return out, cats


def _cb_align(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost 用に Categorical を str に戻す。"""
    X2 = X.copy()
    for c in CAT_FEATURES:
        if c in X2.columns:
            X2[c] = X2[c].astype(str).fillna("UNK")
    return X2


# ===========================================================================
# 回帰アンサンブル
# ===========================================================================
def _train_lgb(X_tr, y_tr, X_te, y_te, w_tr=None) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=63,
        min_data_in_leaf=80, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5,
        random_state=42, verbose=-1)
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
              categorical_feature=CAT_FEATURES)
    return model


def _train_cb(X_tr, y_tr, X_te, y_te, w_tr=None) -> CatBoostRegressor:
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_FEATURES]
    X_tr_cb = _cb_align(X_tr)
    X_te_cb = _cb_align(X_te)
    model = CatBoostRegressor(
        iterations=800, learning_rate=0.05, depth=8,
        l2_leaf_reg=3.0, random_seed=42,
        loss_function="RMSE", eval_metric="MAE",
        early_stopping_rounds=50, verbose=0,
        allow_writing_files=False)
    model.fit(X_tr_cb, y_tr, sample_weight=w_tr,
              cat_features=cat_idx,
              eval_set=(X_te_cb, y_te))
    return model


def _train_xgb(X_tr, y_tr, X_te, y_te, w_tr=None) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=8,
        min_child_weight=20, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42,
        tree_method="hist", enable_categorical=True,
        eval_metric="mae", early_stopping_rounds=50,
        verbosity=0)
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_te, y_te)], verbose=False)
    return model


def train_ensemble(df: pd.DataFrame, target_col: str,
                   feature_cols: list[str],
                   paths: dict[str, Path],
                   cat_categories: dict | None = None,
                   wet_weight: float = 2.0) -> dict:
    work = df.dropna(subset=[target_col]).copy()
    y = work[target_col].astype(float)
    X = work[feature_cols]
    X, cats = _to_categorical(X, CAT_FEATURES, cat_categories)

    # 湿サンプルに wet_weight 倍の重み (sparse condition 補強)
    is_wet = work["is_wet"].fillna(0).astype(int).values
    sample_weight = np.where(is_wet == 1, wet_weight, 1.0)

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, sample_weight,
        test_size=0.1, random_state=42, shuffle=True)
    # eval は重みなしで純MAEを見る (early stoppingは均等評価)
    _ = w_te  # not used

    print(f"  [LGB] training ({target_col}) wet_w={wet_weight} ...")
    lgb_model = _train_lgb(X_tr, y_tr, X_te, y_te, w_tr=w_tr)
    print(f"  [CB ] training ({target_col}) ...")
    cb_model = _train_cb(X_tr, y_tr, X_te, y_te, w_tr=w_tr)
    print(f"  [XGB] training ({target_col}) ...")
    xgb_model = _train_xgb(X_tr, y_tr, X_te, y_te, w_tr=w_tr)

    pred_lgb = lgb_model.predict(X_te)
    pred_cb = cb_model.predict(_cb_align(X_te))
    pred_xgb = xgb_model.predict(X_te)
    pred_avg = (pred_lgb + pred_cb + pred_xgb) / 3.0
    mae_lgb = mean_absolute_error(y_te, pred_lgb)
    mae_cb = mean_absolute_error(y_te, pred_cb)
    mae_xgb = mean_absolute_error(y_te, pred_xgb)
    mae_ens = mean_absolute_error(y_te, pred_avg)
    # 湿サンプル別の MAE
    is_wet_te = work.loc[X_te.index, "is_wet"].fillna(0).astype(int).values
    wet_mask = is_wet_te == 1
    dry_mask = ~wet_mask
    n_wet = int(wet_mask.sum())
    n_dry = int(dry_mask.sum())
    mae_wet = mean_absolute_error(y_te.values[wet_mask],
                                  pred_avg[wet_mask]) if n_wet else None
    mae_dry = mean_absolute_error(y_te.values[dry_mask],
                                  pred_avg[dry_mask]) if n_dry else None
    print(f"  [{target_col}] MAE  LGB={mae_lgb:.4f}  CB={mae_cb:.4f}  "
          f"XGB={mae_xgb:.4f}  ENS={mae_ens:.4f}")
    if mae_wet is not None:
        print(f"    └ ENS wet MAE={mae_wet:.4f} ({n_wet:,} samples)")
    if mae_dry is not None:
        print(f"    └ ENS dry MAE={mae_dry:.4f} ({n_dry:,} samples)")

    joblib.dump(lgb_model, paths["lgb"])
    joblib.dump(cb_model, paths["cb"])
    joblib.dump(xgb_model, paths["xgb"])
    return {"mae_lgb": mae_lgb, "mae_cb": mae_cb,
            "mae_xgb": mae_xgb, "mae_ensemble": mae_ens,
            "mae_wet": mae_wet, "mae_dry": mae_dry,
            "categories": cats}


# ===========================================================================
# ランカー (LGB LambdaRank + XGB Ranker のアンサンブル)
# ===========================================================================
def _build_rank_dataset(df: pd.DataFrame, cat_categories: dict):
    """ランキング学習用にラベル・グループ・特徴量を準備する。"""
    work = df.copy()
    work = work[work["finish_1st"].notna()
                & work["finish_2nd"].notna()
                & work["finish_3rd"].notna()
                & work["bike_no"].notna()
                & work["trial_time"].notna()].copy()
    work["relevance"] = 0
    work.loc[work["bike_no"] == work["finish_1st"], "relevance"] = 4
    work.loc[work["bike_no"] == work["finish_2nd"], "relevance"] = 3
    work.loc[work["bike_no"] == work["finish_3rd"], "relevance"] = 2
    work = work.sort_values(
        ["yyyymmdd", "pid", "race_no", "bike_no"]
    ).reset_index(drop=True)
    grp_sizes = (work.groupby(["yyyymmdd", "pid", "race_no"],
                              sort=False).size().values)
    X = work[RANK_FEATURES]
    X, _ = _to_categorical(X, CAT_FEATURES, cat_categories)
    y = work["relevance"].astype(int).values
    return work, X, y, grp_sizes


def train_ranker(df: pd.DataFrame, cat_categories: dict) -> dict:
    work, X, y, grp_sizes = _build_rank_dataset(df, cat_categories)
    # 時系列分割: 直近10%のレースを valid
    n_valid_races = max(1, int(len(grp_sizes) * 0.1))
    n_valid_rows = int(grp_sizes[-n_valid_races:].sum())
    cut = len(work) - n_valid_rows
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y[:cut], y[cut:]
    grp_tr = grp_sizes[:-n_valid_races]
    grp_te = grp_sizes[-n_valid_races:]
    print(f"  [Ranker] train races={len(grp_tr):,}, "
          f"valid races={len(grp_te):,}")

    # ---- LightGBM LambdaRank ------------------------------------------------
    lgb_ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=1500, learning_rate=0.05,
        num_leaves=63, min_data_in_leaf=80,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        label_gain=[0, 1, 3, 7, 15],
        random_state=42, verbose=-1)
    lgb_ranker.fit(X_tr, y_tr, group=grp_tr,
                   eval_set=[(X_te, y_te)], eval_group=[grp_te],
                   eval_at=[1, 3],
                   callbacks=[lgb.early_stopping(80),
                              lgb.log_evaluation(0)],
                   categorical_feature=CAT_FEATURES)

    # ---- XGBoost Ranker -----------------------------------------------------
    # XGBoost は qid を要求するので連番化
    qid_tr = np.repeat(np.arange(len(grp_tr)), grp_tr)
    qid_te = np.repeat(np.arange(len(grp_te)), grp_te)
    xgb_ranker = XGBRanker(
        objective="rank:pairwise",
        n_estimators=1500, learning_rate=0.05, max_depth=8,
        min_child_weight=20, subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", enable_categorical=True,
        eval_metric="ndcg@3", early_stopping_rounds=80,
        random_state=42, verbosity=0)
    xgb_ranker.fit(X_tr, y_tr, qid=qid_tr,
                   eval_set=[(X_te, y_te)], eval_qid=[qid_te],
                   verbose=False)

    # 評価
    s_lgb = lgb_ranker.predict(X_te)
    s_xgb = xgb_ranker.predict(X_te)
    # 各レース内で z-score 正規化してから平均 (スケール差吸収)
    df_te = work.iloc[cut:].copy()
    df_te["s_lgb"] = s_lgb
    df_te["s_xgb"] = s_xgb
    grp_keys = ["yyyymmdd", "pid", "race_no"]
    for c in ("s_lgb", "s_xgb"):
        df_te[f"{c}_z"] = df_te.groupby(grp_keys)[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    df_te["score_ens"] = (df_te["s_lgb_z"] + df_te["s_xgb_z"]) / 2

    def _hit_metrics(score_col):
        h1 = h3 = total = 0
        for _, g in df_te.groupby(grp_keys, sort=False):
            g = g.sort_values(score_col, ascending=False)
            actual_1st = g["finish_1st"].iloc[0]
            actual_top3 = {g["finish_1st"].iloc[0],
                           g["finish_2nd"].iloc[0],
                           g["finish_3rd"].iloc[0]}
            if g["bike_no"].iloc[0] == actual_1st:
                h1 += 1
            if g["bike_no"].iloc[0] in actual_top3:
                h3 += 1
            total += 1
        return h1 / total, h3 / total, total

    h1_lgb, h3_lgb, total = _hit_metrics("s_lgb")
    h1_xgb, h3_xgb, _ = _hit_metrics("s_xgb")
    h1_ens, h3_ens, _ = _hit_metrics("score_ens")
    print(f"  [Ranker] valid races={total}")
    print(f"    LGB top1={h1_lgb:.3f}  pred1_in_top3={h3_lgb:.3f}")
    print(f"    XGB top1={h1_xgb:.3f}  pred1_in_top3={h3_xgb:.3f}")
    print(f"    ENS top1={h1_ens:.3f}  pred1_in_top3={h3_ens:.3f}")

    joblib.dump(lgb_ranker, RANKER_LGB_PATH)
    joblib.dump(xgb_ranker, RANKER_XGB_PATH)
    return {"lgb_top1": h1_lgb, "xgb_top1": h1_xgb, "ens_top1": h1_ens,
            "lgb_pred1_top3": h3_lgb, "xgb_pred1_top3": h3_xgb,
            "ens_pred1_top3": h3_ens, "n_valid_races": total}


# ===========================================================================
# 全体オーケストレーション
# ===========================================================================
def full_retrain() -> dict:
    print("[info] loading training data from DuckDB ...")
    df = load_training_data()
    print(f"[info] rows = {len(df):,}")
    df = prepare_features(df)

    print("[info] training trial_time ENSEMBLE (no stacking) ...")
    res_trial = train_ensemble(
        df, "trial_time", TRIAL_FEATURES,
        {"lgb": TRIAL_LGB_PATH, "cb": TRIAL_CB_PATH,
         "xgb": TRIAL_XGB_PATH})
    cats = res_trial["categories"]

    print("[info] training expected_race_time ENSEMBLE "
          "(with trial_time stacking) ...")
    res_race = train_ensemble(
        df, "expected_race_time", RACE_FEATURES,
        {"lgb": RACE_LGB_PATH, "cb": RACE_CB_PATH,
         "xgb": RACE_XGB_PATH},
        cat_categories=cats)

    print("[info] training Ranker ENSEMBLE (LGB + XGB) ...")
    res_rank = train_ranker(df, cats)

    meta = {
        "trial_features": TRIAL_FEATURES,
        "race_features": RACE_FEATURES,
        "rank_features": RANK_FEATURES,
        "base_num_features": BASE_NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "categories": {k: list(v) for k, v in cats.items()},
        "global_mean_trial": df.attrs.get("gmt"),
        "global_mean_race": df.attrs.get("gmr"),
        "smoothing_k": {"player": K_PLAYER, "bike": K_BIKE,
                        "bucket": K_BUCKET},
    }
    joblib.dump(meta, META_PATH)
    print(f"[done] saved -> {MODEL_DIR}")
    return {"mode": "full", "n_train": len(df),
            "trial": {k: v for k, v in res_trial.items()
                      if k != "categories"},
            "race": res_race, "ranker": res_rank}


# ===========================================================================
# 増分学習 (LightGBM のみ warm-start)
# ===========================================================================
def incremental_update(new_df: pd.DataFrame, additional_rounds: int = 80,
                       learning_rate: float = 0.02) -> dict:
    if not META_PATH.exists():
        raise FileNotFoundError("先に全学習を実行してください (full_retrain)。")
    meta = joblib.load(META_PATH)
    categories = {k: pd.Index(v) for k, v in meta["categories"].items()}

    new_df = prepare_features(new_df, global_means={
        "trial": meta.get("global_mean_trial") or 3.5,
        "race": meta.get("global_mean_race") or 3.55,
    })
    results: dict[str, dict] = {}

    for target_col, model_path, feats in [
        ("trial_time", TRIAL_LGB_PATH, TRIAL_FEATURES),
        ("expected_race_time", RACE_LGB_PATH, RACE_FEATURES),
    ]:
        work = new_df.dropna(subset=[target_col]).copy()
        if len(work) < 5:
            results[target_col] = {"skipped": True, "n_samples": len(work),
                                   "reason": "n<5"}
            continue
        y = work[target_col].astype(float)
        X = work[feats]
        X, _ = _to_categorical(X, CAT_FEATURES, categories)
        existing = joblib.load(model_path)
        params = existing.get_params()
        new_model = lgb.LGBMRegressor(
            n_estimators=additional_rounds, learning_rate=learning_rate,
            num_leaves=params.get("num_leaves", 63),
            min_data_in_leaf=max(5, min(20, len(work) // 5)),
            feature_fraction=params.get("feature_fraction", 0.9),
            bagging_fraction=params.get("bagging_fraction", 0.9),
            bagging_freq=params.get("bagging_freq", 5),
            random_state=42, verbose=-1)
        new_model.fit(X, y, init_model=existing,
                      categorical_feature=CAT_FEATURES)
        joblib.dump(new_model, model_path)
        results[target_col] = {"skipped": False, "n_samples": len(work),
                               "added_rounds": additional_rounds,
                               "learning_rate": learning_rate}
    results["note"] = ("LGB のみ warm-start 更新。CB/XGB/Ranker は "
                       "「全データで再学習」で反映してください。")
    return results


def main() -> int:
    full_retrain()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
