"""Streamlit GUI: オートレース 試走タイム & 着順 予測。

起動:
    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from db_manager import (
    init_database, upsert_race_rows, load_rows_for_keys,
    get_imported_race_dates, get_latest_imported_date,
)
from entry_scraper import fetch_entry, build_url
from predictor import Predictor
from result_fetcher import fetch_race_result
from train_model import full_retrain, incremental_update

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
SCHEDULE_CSV = BASE_DIR / "autorace_schedule.csv"

PID_MAP = {
    "川口": 2,
    "伊勢崎": 3,
    "浜松": 4,
    "飯塚": 5,
    "山陽": 6,
}
WEATHERS = ["晴", "曇", "雨", "雪", "霧"]
TRACKS = ["良走路", "稍湿走路", "湿走路", "重走路", "不良走路"]

st.set_page_config(page_title="オートレース予想", page_icon=":motorcycle:",
                   layout="wide")


@st.cache_resource
def load_predictor() -> Predictor | None:
    required = ["trial_time_lgb.pkl", "trial_time_cb.pkl",
                "trial_time_xgb.pkl", "race_time_lgb.pkl",
                "race_time_cb.pkl", "race_time_xgb.pkl",
                "finish_ranker_lgb.pkl", "finish_ranker_xgb.pkl",
                "feature_meta.pkl"]
    for f in required:
        if not (MODEL_DIR / f).exists():
            return None
    return Predictor(MODEL_DIR)


@st.cache_resource
def ensure_db_ready() -> bool:
    init_database(force=False)
    return True


@st.cache_data(ttl=600)
def _load_schedule_index() -> dict[str, set[str]]:
    """autorace_schedule.csv を読み込み {yyyymmdd_str: {racetrack,...}} に変換。

    範囲外の日付は空辞書に含まれない (= 自動検出時に「全5会場を候補」扱い)。
    """
    if not SCHEDULE_CSV.exists():
        return {}
    df = pd.read_csv(SCHEDULE_CSV)
    df = df.dropna(subset=["date", "racetrack"])
    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        try:
            d = pd.to_datetime(str(r["date"]).strip(), errors="coerce")
        except Exception:
            continue
        if pd.isna(d):
            continue
        track = str(r["racetrack"]).strip()
        # "川口2" は pid=2 と同じなので "川口" として扱う (DB側でも統合済み)
        if track == "川口2":
            track = "川口"
        if track not in PID_MAP:
            continue
        out.setdefault(d.strftime("%Y%m%d"), set()).add(track)
    return out


def detect_missing_races(date_from: dt.date, date_to: dt.date,
                         force: bool = False) -> list[tuple[str, int, str]]:
    """未取り込みレースの候補 (yyyymmdd, pid, racetrack) を返す。

    - スケジュールCSVに記載がある日付は、その日に開催のある会場のみ対象。
    - スケジュール範囲外の日付は、5会場全てを対象 (実際の有無は HTTP で判定)。
    - force=True の場合、DBに既存の (yyyymmdd, pid) も含める。
    """
    imported = get_imported_race_dates()
    sched = _load_schedule_index()
    sched_dates = set(sched.keys())
    sched_min = min(sched_dates) if sched_dates else None
    sched_max = max(sched_dates) if sched_dates else None

    candidates: list[tuple[str, int, str]] = []
    d = date_from
    while d <= date_to:
        ymd_str = d.strftime("%Y%m%d")
        ymd_int = int(ymd_str)

        if sched_min is not None and sched_min <= ymd_str <= sched_max:
            # スケジュールCSV範囲内 → 載っている会場のみ候補
            tracks = sched.get(ymd_str, set())
        else:
            # 範囲外 (主に最近の日付) → 5会場全て試行
            tracks = set(PID_MAP.keys())

        for track in tracks:
            pid = PID_MAP[track]
            if not force and ymd_int in imported.get(pid, set()):
                continue
            candidates.append((ymd_str, pid, track))
        d += dt.timedelta(days=1)
    return candidates


def render_sidebar() -> dict:
    st.sidebar.header("レース指定")
    track = st.sidebar.selectbox("会場", list(PID_MAP.keys()), index=4)
    pid = PID_MAP[track]
    rdt_date = st.sidebar.date_input("開催日", value=dt.date.today())
    rno = st.sidebar.number_input("レース番号", min_value=1, max_value=12,
                                  value=1, step=1)

    st.sidebar.markdown("---")
    st.sidebar.header("天候 (空欄なら自動補完)")
    weather = st.sidebar.selectbox("天候", [""] + WEATHERS, index=0)
    track_cond = st.sidebar.selectbox("走路状況", [""] + TRACKS, index=0)
    track_temp = st.sidebar.text_input("走路温度 (℃)", value="")
    air_temp = st.sidebar.text_input("気温 (℃)", value="")
    humidity = st.sidebar.text_input("湿度 (%)", value="")

    def _f(v: str):
        v = v.strip()
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    return {
        "racetrack": track,
        "pid": pid,
        "yyyymmdd": rdt_date.strftime("%Y%m%d"),
        "race_no": int(rno),
        "weather": weather or None,
        "track_condition": track_cond or None,
        "track_temp_c": _f(track_temp),
        "air_temp_c": _f(air_temp),
        "humidity_pct": _f(humidity),
    }


def render_predict_tab(predictor: Predictor, cfg: dict) -> None:
    """予測タブ。"""
    url = build_url(cfg["pid"], cfg["yyyymmdd"], cfg["race_no"])

    st.markdown(f"**対象レース**: {cfg['racetrack']} "
                f"{cfg['yyyymmdd']} {cfg['race_no']}R")
    st.markdown(f"[Gamboo 出走表ページを開く]({url})  "
                f"(取れない場合は公開ページ`/autorace/program?pt=8`に自動フォールバック)")

    col1, _ = st.columns([1, 4])
    with col1:
        run = st.button(":mag: 出走表取得 & 予測", type="primary",
                        use_container_width=True, key="predict_btn")

    if not run:
        st.info("左サイドバーで条件を指定して「予測」を押してください。")
        return

    with st.spinner("出走表を取得中..."):
        try:
            entry = fetch_entry(cfg["pid"], cfg["yyyymmdd"], cfg["race_no"])
        except Exception as e:
            st.error(f"出走表取得失敗: {e}")
            return

    if not entry["players"]:
        st.warning("出走情報が取得できませんでした (中止 / 未開催 の可能性)。")
        return

    if entry.get("source_url") and entry["source_url"] != url:
        st.caption(f"出走表取得元: {entry['source_url']} (フォールバック)")

    race_info = {
        **entry,
        "weather": cfg["weather"] or entry.get("weather"),
        "track_condition": cfg["track_condition"]
                          or entry.get("track_condition"),
        "track_temp_c": cfg["track_temp_c"]
                       if cfg["track_temp_c"] is not None
                       else entry.get("track_temp_c"),
        "air_temp_c": cfg["air_temp_c"]
                     if cfg["air_temp_c"] is not None
                     else entry.get("air_temp_c"),
        "humidity_pct": cfg["humidity_pct"]
                       if cfg["humidity_pct"] is not None
                       else entry.get("humidity_pct"),
    }

    st.subheader("出走表")
    entry_df = pd.DataFrame(entry["players"])
    st.dataframe(entry_df, use_container_width=True, hide_index=True)

    with st.spinner("予測中..."):
        result = predictor.predict_race(race_info, entry["players"])

    used = result.attrs.get("used_defaults", {})
    st.subheader("レース条件")
    wcols = st.columns(5)
    src_map = {"天候": "weather", "走路状況": "track_condition",
               "走路温度": "track_temp_c", "気温": "air_temp_c",
               "湿度": "humidity_pct"}
    weather_view = {k: race_info.get(v) for k, v in src_map.items()}
    for i, (k, v) in enumerate(weather_view.items()):
        label = f"{k} {'(自動補完)' if src_map[k] in used else ''}"
        wcols[i].metric(label, value=str(v))

    st.subheader(":dart: 予測結果")
    if (result["trial_time_source"] == "実測").any():
        n_actual = (result["trial_time_source"] == "実測").sum()
        st.caption(f":white_check_mark: 試走T 実測値を {n_actual}/"
                   f"{len(result)} 車で stacking に利用 (精度向上)")
    show = result[["bike_no", "player_name", "rank", "handicap_m",
                   "pred_trial_time", "actual_trial_time",
                   "trial_time_source",
                   "pred_race_time", "rank_score",
                   "pred_finish_order",
                   "pred_finish_order_by_time"]].copy()
    show["actual_trial_time"] = show["actual_trial_time"].apply(
        lambda v: "-" if (v is None or pd.isna(v)) else v)
    show = show.rename(columns={
        "bike_no": "車番", "player_name": "選手名", "rank": "ランク",
        "handicap_m": "ハンデ(m)",
        "pred_trial_time": "予測試走T",
        "actual_trial_time": "実測試走T",
        "trial_time_source": "T出典",
        "pred_race_time": "予測競走T",
        "rank_score": "ランカースコア",
        "pred_finish_order": "予測着順(Ranker)",
        "pred_finish_order_by_time": "予測着順(競走T順)",
    })
    st.markdown("**車番順**")
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.markdown("**予測着順順 (Ranker)**")
    ordered = show.sort_values("予測着順(Ranker)").reset_index(drop=True)
    st.dataframe(ordered, use_container_width=True, hide_index=True)

    st.subheader(":trophy: 予想印")
    top3 = ordered.head(3)
    summary_cols = st.columns(3)
    marks = ["◎ 本命", "○ 対抗", "▲ 単穴"]
    for i, (_, r) in enumerate(top3.iterrows()):
        summary_cols[i].metric(
            marks[i],
            f"{int(r['車番'])}号 {r['選手名']}",
            f"予測競走T {r['予測競走T']} / score {r['ランカースコア']}")


def _run_auto_ingest(date_from: dt.date, date_to: dt.date,
                     force: bool, retrain_after: bool) -> None:
    """自動検出 → 取得 → upsert → 全再学習 を一括実行する。"""
    candidates = detect_missing_races(date_from, date_to, force=force)
    if not candidates:
        st.success("未取り込みレースは見つかりませんでした。DBは最新です。")
        return

    st.info(f"候補: 未取り込み (日付 × 会場) {len(candidates)} 件を順に試行します。")
    progress = st.progress(0.0)
    log = st.empty()

    total_inserted = 0
    total_replaced = 0
    all_race_keys: list[tuple] = []
    n_total = len(candidates)
    n_event_hit = 0
    n_no_event = 0

    for i, (ymd, pid, track) in enumerate(candidates, start=1):
        # まず R=1 を試して、その日に開催があるか確認
        try:
            r1 = fetch_race_result(pid, ymd, 1)
        except Exception as e:
            log.write(f"[err] {ymd} {track} 1R: {e}")
            progress.progress(i / n_total)
            continue
        if r1["status"] != 200 or not r1["rows"] or not r1["has_result"]:
            n_no_event += 1
            log.write(f"[skip] {ymd} {track}: 開催なし or 結果未確定")
            progress.progress(i / n_total)
            continue

        # 開催ありと判明 → R=1 を upsert、R=2..12 を順に
        up = upsert_race_rows(r1["rows"])
        total_inserted += up["inserted"]
        total_replaced += up["replaced"]
        all_race_keys.extend(up["races_updated"])
        last_rno = 1
        for rno in range(2, 13):
            try:
                res = fetch_race_result(pid, ymd, rno)
            except Exception as e:
                log.write(f"[err] {ymd} {track} {rno}R: {e}")
                continue
            if res["status"] != 200 or not res["rows"] or not res["has_result"]:
                break
            up = upsert_race_rows(res["rows"])
            total_inserted += up["inserted"]
            total_replaced += up["replaced"]
            all_race_keys.extend(up["races_updated"])
            last_rno = rno
        n_event_hit += 1
        log.write(f"[ok] {ymd} {track}: 1R〜{last_rno}R を取得 ・ "
                  f"insert={total_inserted}, replace={total_replaced}")
        progress.progress(i / n_total)

    # 重複除去
    seen = set()
    uniq_keys = []
    for k in all_race_keys:
        tup = (int(k[0]), int(k[1]), int(k[2]))
        if tup not in seen:
            seen.add(tup)
            uniq_keys.append(tup)

    st.success(
        f"自動取得完了: 開催あり {n_event_hit} / 開催なし {n_no_event}, "
        f"insert {total_inserted} 行, replace {total_replaced} 行, "
        f"対象レース {len(uniq_keys)} 件")
    st.session_state["last_updated_keys"] = uniq_keys

    if retrain_after and uniq_keys:
        st.markdown("---")
        st.markdown("**:brain: 全データ再学習を実行中...**")
        with st.spinner("全データ再学習中 (1分程度)..."):
            res = full_retrain()
        st.success("再学習完了")
        st.json(res)
        load_predictor.clear()
        st.info("予測タブのモデルは次の予測実行から自動反映されます。")
    elif retrain_after and not uniq_keys:
        st.caption("新規取り込みがなかったため再学習はスキップしました。")


def render_result_tab() -> None:
    """結果取得→DB更新→モデル補正タブ。"""
    st.subheader(":arrows_counterclockwise: 実績更新 (結果取得 → DB → モデル補正)")
    st.caption("完走したレースの結果を Gamboo から取得し、DuckDB に upsert "
               "した上で予測モデルを補正/再学習します。")

    # === 自動取得セクション =================================================
    st.markdown("### :robot_face: 未取り込みレースを自動検出 → 取得 → 再学習")
    latest_ymd = get_latest_imported_date()
    if latest_ymd:
        latest_d = dt.datetime.strptime(str(latest_ymd), "%Y%m%d").date()
        st.caption(f"DB の最新取り込み日: **{latest_d.isoformat()}** "
                   f"(全 {len(get_imported_race_dates())} pid 集計)")
        default_from = latest_d + dt.timedelta(days=1)
    else:
        st.caption("DBが空です。最初の取り込みを実行します。")
        default_from = dt.date.today() - dt.timedelta(days=14)
    default_to = dt.date.today() - dt.timedelta(days=1)

    with st.expander(":wrench: 自動取得オプション", expanded=True):
        a1, a2 = st.columns(2)
        auto_from = a1.date_input("検出開始日", value=default_from,
                                  key="auto_from")
        auto_to = a2.date_input("検出終了日 (=昨日が安全)",
                                value=default_to, key="auto_to")
        force = st.checkbox(
            "DBに既存の日付も再取得 (上書きしたい場合のみ)",
            value=False, key="auto_force")
        retrain_after = st.checkbox(
            ":brain: 取得完了後に全データ再学習を自動実行",
            value=True, key="auto_retrain")
        c_prev, c_run = st.columns([1, 2])
        with c_prev:
            preview = st.button(":mag: 検出のみ (プレビュー)",
                                key="auto_preview",
                                use_container_width=True)
        with c_run:
            run_auto = st.button(
                ":magic_wand: 自動検出 → 取得 → 再学習を実行",
                type="primary", key="auto_run",
                use_container_width=True)

        if preview:
            if auto_to < auto_from:
                st.error("検出終了日は開始日以降を指定してください。")
            else:
                cands = detect_missing_races(auto_from, auto_to, force=force)
                if not cands:
                    st.success("未取り込みレースは見つかりませんでした。")
                else:
                    st.info(f"検出: {len(cands)} 件の (日付 × 会場) が未取り込みです。")
                    prev_df = pd.DataFrame(
                        cands, columns=["yyyymmdd", "pid", "会場"])
                    st.dataframe(prev_df, use_container_width=True,
                                 hide_index=True, height=240)

        if run_auto:
            if auto_to < auto_from:
                st.error("検出終了日は開始日以降を指定してください。")
            else:
                _run_auto_ingest(auto_from, auto_to, force, retrain_after)

    st.divider()
    st.markdown("### :pushpin: 手動取得 (会場・日付・最大R を指定)")
    with st.expander("対象レース指定", expanded=False):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        track = c1.selectbox("会場", list(PID_MAP.keys()), index=0,
                             key="res_track")
        pid = PID_MAP[track]
        date_from = c2.date_input("日付 from",
                                  value=dt.date.today() - dt.timedelta(days=1),
                                  key="res_from")
        date_to = c3.date_input("日付 to",
                                value=dt.date.today() - dt.timedelta(days=1),
                                key="res_to")
        max_rno = c4.number_input("最大R", min_value=1, max_value=12,
                                  value=12, step=1, key="res_max_rno")

    fetch_btn = st.button(":satellite_antenna: 結果を取得して DB に保存",
                          type="primary", key="fetch_results")
    if fetch_btn:
        if date_to < date_from:
            st.error("日付の to は from 以降を指定してください。")
        else:
            days = (date_to - date_from).days + 1
            total_inserted = 0
            total_replaced = 0
            all_race_keys: list[tuple] = []
            progress = st.progress(0.0)
            log = st.empty()
            n_total = days * int(max_rno)
            i = 0
            for d_offset in range(days):
                d = date_from + dt.timedelta(days=d_offset)
                ymd = d.strftime("%Y%m%d")
                for rno in range(1, int(max_rno) + 1):
                    i += 1
                    try:
                        res = fetch_race_result(pid, ymd, rno)
                    except Exception as e:
                        log.write(f"[err] {ymd} {rno}R: {e}")
                        progress.progress(i / n_total)
                        continue
                    if res["status"] != 200 or not res["rows"]:
                        log.write(f"[skip] {ymd} {rno}R (status={res['status']})")
                        progress.progress(i / n_total)
                        continue
                    if not res["has_result"]:
                        log.write(f"[skip] {ymd} {rno}R: 結果未確定")
                        progress.progress(i / n_total)
                        continue
                    up = upsert_race_rows(res["rows"])
                    total_inserted += up["inserted"]
                    total_replaced += up["replaced"]
                    all_race_keys.extend(up["races_updated"])
                    log.write(
                        f"[ok] {ymd} {rno}R: "
                        f"insert={up['inserted']}, replace={up['replaced']}"
                    )
                    progress.progress(i / n_total)
            st.success(
                f"完了: insert {total_inserted} 行, replace {total_replaced} 行, "
                f"対象レース {len(all_race_keys)}件")
            st.session_state["last_updated_keys"] = all_race_keys

    st.divider()
    st.subheader(":brain: モデル更新")
    last_keys = st.session_state.get("last_updated_keys", [])
    if last_keys:
        st.caption(f"直近で DB に取り込まれたレース: {len(last_keys)} 件")
    else:
        st.caption("(まだ DB 取り込みを実行していません)")

    cA, cB = st.columns(2)
    with cA:
        rounds = st.number_input("追加ブースト回数", min_value=10,
                                 max_value=500, value=80, step=10,
                                 key="inc_rounds")
        lr = st.number_input("学習率", min_value=0.001, max_value=0.2,
                             value=0.02, step=0.005, format="%.3f",
                             key="inc_lr")
        if st.button(":zap: 補正 (増分学習)",
                     disabled=not last_keys, key="inc_btn",
                     use_container_width=True):
            with st.spinner("増分学習中..."):
                new_df = load_rows_for_keys(last_keys)
                if new_df.empty:
                    st.warning("対象データが空です。")
                else:
                    res = incremental_update(
                        new_df, additional_rounds=int(rounds),
                        learning_rate=float(lr))
                    st.success("増分学習完了")
                    st.json(res)
                    load_predictor.clear()  # モデルキャッシュ破棄
                    st.info("予測タブのモデルは次の予測実行から自動反映されます。")
    with cB:
        st.markdown("**全データで再学習**")
        st.caption("既存モデルを破棄して、現在のDB全件で再構築します (1分程度)。")
        if st.button(":arrows_counterclockwise: 全データで再学習",
                     key="full_btn", use_container_width=True):
            with st.spinner("全データ再学習中..."):
                res = full_retrain()
                st.success("再学習完了")
                st.json(res)
                load_predictor.clear()


def main():
    st.title(":motorcycle: オートレース 試走タイム / 着順予想")
    st.caption("supacon_dataset.csv + 選手特長一覧表.csv を機械学習で学習")

    ensure_db_ready()
    predictor = load_predictor()
    if predictor is None:
        st.error("学習済みモデルが見つかりません。先にターミナルで以下を実行してください:")
        st.code("python train_model.py", language="bash")
        st.stop()

    cfg = render_sidebar()

    tab_pred, tab_update = st.tabs(
        [":dart: 予測", ":arrows_counterclockwise: 結果取込/モデル更新"])
    with tab_pred:
        render_predict_tab(predictor, cfg)
    with tab_update:
        render_result_tab()


if __name__ == "__main__":
    main()
