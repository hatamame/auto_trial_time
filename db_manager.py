"""DuckDB によるデータ管理モジュール。

supacon_dataset.csv と 選手特長一覧表.csv を DuckDB に取り込み、
学習・予測で参照できる形で提供する。
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import duckdb
import pandas as pd


def normalize_player_name(name: str | None) -> str:
    """選手名を正規化する。

    - NFKC で全角/半角の互換正規化
    - 全角・半角スペース、記号、改行を除去
    - 大文字小文字を同一視 (英字混入対策)
    """
    if name is None:
        return ""
    s = unicodedata.normalize("NFKC", str(name))
    for ch in (" ", "　", "\t", "\n", "\r", "・", ".", "．"):
        s = s.replace(ch, "")
    return s.casefold()

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "autorace.duckdb"
SUPACON_CSV = BASE_DIR / "supacon_dataset.csv"
PLAYER_FEATURE_CSV = BASE_DIR / "選手特長一覧表.csv"

# 数値・カテゴリ系の主要列
SUPACON_COLUMNS = [
    "date", "yyyymmdd", "racetrack", "pid", "race_no",
    "grade", "event_type",
    "weather", "track_condition", "track_temp_c", "air_temp_c", "humidity_pct",
    "bike_no", "rank", "player_lg", "license_period",
    "player_name", "bike_class", "bike_name", "handicap_m",
    "trial_time", "expected_race_time",
    "finish_1st", "finish_2nd", "finish_3rd", "exacta_payout_yen",
]


def get_connection(db_path: Path | str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """DuckDB 接続を取得する (Python UDF を登録済み)。"""
    con = duckdb.connect(str(db_path))
    # 既存登録があると上書きエラーになるため try/except で吸収
    try:
        con.create_function("norm_name", normalize_player_name,
                            ["VARCHAR"], "VARCHAR")
    except (duckdb.CatalogException, duckdb.InvalidInputException):
        pass
    return con


def init_database(force: bool = False) -> None:
    """CSV を読み込んでテーブル化する。

    force=True で既存テーブルを破棄して再生成。
    """
    con = get_connection()
    try:
        if force:
            con.execute("DROP TABLE IF EXISTS races;")
            con.execute("DROP TABLE IF EXISTS player_features;")

        # races テーブル
        existing = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'races';"
        ).fetchone()[0]
        if existing == 0:
            con.execute(f"""
                CREATE TABLE races AS
                SELECT * FROM read_csv_auto('{SUPACON_CSV.as_posix()}',
                                            header=True, sample_size=-1);
            """)
            # 着順を選手単位の列に展開
            con.execute("""
                ALTER TABLE races ADD COLUMN IF NOT EXISTS finish_position INTEGER;
            """)
            con.execute("""
                UPDATE races SET finish_position = CASE
                    WHEN bike_no = finish_1st THEN 1
                    WHEN bike_no = finish_2nd THEN 2
                    WHEN bike_no = finish_3rd THEN 3
                    ELSE NULL
                END;
            """)

        # player_features テーブル (正規化名で索引)
        existing_pf = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'player_features';"
        ).fetchone()[0]
        if existing_pf == 0:
            con.execute(f"""
                CREATE TABLE player_features AS
                SELECT
                    norm_name(選手名) AS player_name,
                    選手名 AS player_name_raw,
                    "S力" AS s_power,
                    独走力 AS solo_power,
                    追い込み力 AS chase_power,
                    雨巧拙 AS rain_skill,
                    コース AS course_pref
                FROM read_csv_auto('{PLAYER_FEATURE_CSV.as_posix()}',
                                   header=True);
            """)
    finally:
        con.close()


def load_training_data() -> pd.DataFrame:
    """学習用 DataFrame を返す (races と player_features を結合)。"""
    init_database()
    con = get_connection()
    try:
        df = con.execute("""
            SELECT
                r.*,
                pf.s_power, pf.solo_power, pf.chase_power,
                pf.rain_skill, pf.course_pref
            FROM races r
            LEFT JOIN player_features pf
              ON norm_name(r.player_name) = pf.player_name
            WHERE r.bike_no IS NOT NULL
        """).df()
        return df
    finally:
        con.close()


def get_player_recent_stats(player_name: str, n: int = 10) -> dict:
    """直近 n レースの集計を返す (予測時のフォーム特徴量)。"""
    con = get_connection()
    try:
        key = normalize_player_name(player_name)
        df = con.execute("""
            SELECT trial_time, expected_race_time, finish_position
            FROM races
            WHERE norm_name(player_name) = ?
              AND trial_time IS NOT NULL
            ORDER BY yyyymmdd DESC, race_no DESC
            LIMIT ?
        """, [key, n]).df()
        if df.empty:
            return {"recent_trial_avg": None, "recent_race_avg": None,
                    "recent_top3_rate": None, "recent_n": 0}
        return {
            "recent_trial_avg": float(df["trial_time"].mean()),
            "recent_race_avg": float(df["expected_race_time"].mean()),
            "recent_top3_rate": float(
                (df["finish_position"].fillna(99) <= 3).mean()),
            "recent_n": int(len(df)),
        }
    finally:
        con.close()


_WET_WHERE = """
    (weather LIKE '%雨%' OR weather LIKE '%雪%'
     OR track_condition LIKE '%湿%' OR track_condition LIKE '%斑%')
"""


def get_player_interaction_stats(player_name: str, racetrack: str) -> dict:
    """選手 × 会場 / 選手 × 湿 の過去平均を取得 (予測時用)。"""
    con = get_connection()
    try:
        key = normalize_player_name(player_name)
        row = con.execute("""
            SELECT AVG(trial_time), AVG(expected_race_time), COUNT(*)
            FROM races
            WHERE norm_name(player_name) = ?
              AND racetrack = ?
              AND trial_time IS NOT NULL
        """, [key, racetrack]).fetchone()
        wet_row = con.execute(f"""
            SELECT AVG(trial_time), AVG(expected_race_time), COUNT(*)
            FROM races
            WHERE norm_name(player_name) = ?
              AND {_WET_WHERE}
              AND trial_time IS NOT NULL
        """, [key]).fetchone()
        # 直近5走の湿レース
        recent_wet = con.execute(f"""
            SELECT AVG(trial_time), AVG(expected_race_time), COUNT(*)
            FROM (
                SELECT trial_time, expected_race_time
                FROM races
                WHERE norm_name(player_name) = ?
                  AND {_WET_WHERE}
                  AND trial_time IS NOT NULL
                ORDER BY yyyymmdd DESC, race_no DESC
                LIMIT 5
            )
        """, [key]).fetchone()
        return {
            "player_at_track_trial_avg": row[0],
            "player_at_track_race_avg": row[1],
            "player_at_track_n": int(row[2]) if row[2] else 0,
            "player_in_wet_trial_avg": wet_row[0],
            "player_in_wet_race_avg": wet_row[1],
            "player_in_wet_n": int(wet_row[2]) if wet_row[2] else 0,
            "recent5_wet_trial_avg": recent_wet[0],
            "recent5_wet_race_avg": recent_wet[1],
            "recent5_wet_n": int(recent_wet[2]) if recent_wet[2] else 0,
        }
    finally:
        con.close()


def get_bike_wet_stats(bike_name: str | None) -> dict:
    """車両 × 湿 の過去集計。"""
    if not bike_name:
        return {"avg_trial": None, "avg_race": None, "n": 0}
    con = get_connection()
    try:
        row = con.execute(f"""
            SELECT AVG(trial_time), AVG(expected_race_time), COUNT(*)
            FROM races
            WHERE bike_name = ?
              AND {_WET_WHERE}
              AND trial_time IS NOT NULL
        """, [str(bike_name)]).fetchone()
        return {
            "avg_trial": row[0],
            "avg_race": row[1],
            "n": int(row[2]) if row[2] else 0,
        }
    finally:
        con.close()


def get_player_cond_stats(player_name: str, track_condition: str) -> dict:
    """選手 × 走路状況 の過去集計 (target encoding 用)。"""
    con = get_connection()
    try:
        key = normalize_player_name(player_name)
        row = con.execute("""
            SELECT AVG(trial_time), AVG(expected_race_time), COUNT(*)
            FROM races
            WHERE norm_name(player_name) = ?
              AND COALESCE(track_condition, 'UNK') = ?
              AND trial_time IS NOT NULL
        """, [key, track_condition or "UNK"]).fetchone()
        return {
            "avg_trial": row[0],
            "avg_race": row[1],
            "n": int(row[2]) if row[2] else 0,
        }
    finally:
        con.close()


def get_player_stats(player_name: str) -> dict:
    """選手の過去統計 (平均試走タイム等) を返す。"""
    con = get_connection()
    try:
        row = con.execute("""
            SELECT
                COUNT(*) AS n_races,
                AVG(trial_time) AS avg_trial_time,
                AVG(expected_race_time) AS avg_race_time,
                AVG(CASE WHEN finish_position = 1 THEN 1.0 ELSE 0.0 END)
                    AS win_rate,
                AVG(CASE WHEN finish_position <= 3 THEN 1.0 ELSE 0.0 END)
                    AS top3_rate
            FROM races
            WHERE norm_name(player_name) = ?
              AND trial_time IS NOT NULL
        """, [normalize_player_name(player_name)]).fetchone()
        if row is None or row[0] == 0:
            return {"n_races": 0, "avg_trial_time": None,
                    "avg_race_time": None, "win_rate": None,
                    "top3_rate": None}
        return {"n_races": int(row[0]),
                "avg_trial_time": row[1],
                "avg_race_time": row[2],
                "win_rate": row[3],
                "top3_rate": row[4]}
    finally:
        con.close()


def get_player_features(player_name: str) -> dict | None:
    """選手特長一覧から特徴量を取得。"""
    con = get_connection()
    try:
        row = con.execute("""
            SELECT s_power, solo_power, chase_power, rain_skill, course_pref
            FROM player_features
            WHERE player_name = ?
        """, [normalize_player_name(player_name)]).fetchone()
        if row is None:
            return None
        return {"s_power": row[0], "solo_power": row[1],
                "chase_power": row[2], "rain_skill": row[3],
                "course_pref": row[4]}
    finally:
        con.close()


def get_bike_stats(bike_name: str | None) -> dict:
    """車両 (bike_name) 別の過去集計を返す。"""
    if not bike_name:
        return {"avg_trial": None, "avg_race": None, "n": 0}
    con = get_connection()
    try:
        row = con.execute("""
            SELECT AVG(trial_time), AVG(expected_race_time), COUNT(*)
            FROM races
            WHERE bike_name = ?
              AND trial_time IS NOT NULL
        """, [str(bike_name)]).fetchone()
        return {
            "avg_trial": row[0],
            "avg_race": row[1],
            "n": int(row[2]) if row[2] else 0,
        }
    finally:
        con.close()


def get_bucket_stats(racetrack: str, track_temp_bucket: str,
                     weather: str | None) -> dict:
    """バケット帯 × 会場 × 天候 の過去集計を返す。"""
    con = get_connection()
    try:
        # 同じバケット計算ロジックを SQL 側で再現
        bucket_sql = """
            CASE
                WHEN track_temp_c IS NULL THEN 'UNK'
                WHEN track_temp_c < 20 THEN 'cold'
                WHEN track_temp_c < 35 THEN 'cool'
                WHEN track_temp_c < 50 THEN 'warm'
                ELSE 'hot'
            END
        """
        row = con.execute(f"""
            SELECT AVG(trial_time), AVG(expected_race_time), COUNT(*)
            FROM races
            WHERE racetrack = ?
              AND ({bucket_sql}) = ?
              AND COALESCE(weather, 'UNK') = ?
              AND trial_time IS NOT NULL
        """, [racetrack, track_temp_bucket,
              weather if weather else "UNK"]).fetchone()
        return {
            "avg_trial": row[0],
            "avg_race": row[1],
            "n": int(row[2]) if row[2] else 0,
        }
    finally:
        con.close()


def get_global_means() -> dict:
    """trial_time / expected_race_time の全体平均を返す。"""
    con = get_connection()
    try:
        row = con.execute("""
            SELECT AVG(trial_time), AVG(expected_race_time)
            FROM races
            WHERE trial_time IS NOT NULL
        """).fetchone()
        return {"global_mean_trial": row[0] or 3.5,
                "global_mean_race": row[1] or 3.55}
    finally:
        con.close()


def get_weather_defaults(racetrack: str, month: int) -> dict:
    """天候等のデフォルト値を、同会場・同月の平均から算出する。"""
    con = get_connection()
    try:
        row = con.execute("""
            SELECT
                AVG(track_temp_c) AS track_temp_c,
                AVG(air_temp_c) AS air_temp_c,
                AVG(humidity_pct) AS humidity_pct,
                MODE(weather) AS weather,
                MODE(track_condition) AS track_condition
            FROM races
            WHERE racetrack = ?
              AND CAST(SUBSTR(CAST(yyyymmdd AS VARCHAR), 5, 2) AS INTEGER) = ?
              AND weather IS NOT NULL
        """, [racetrack, month]).fetchone()
        if row is None or row[0] is None:
            # フォールバック (会場全期間平均)
            row = con.execute("""
                SELECT
                    AVG(track_temp_c) AS track_temp_c,
                    AVG(air_temp_c) AS air_temp_c,
                    AVG(humidity_pct) AS humidity_pct,
                    MODE(weather) AS weather,
                    MODE(track_condition) AS track_condition
                FROM races WHERE racetrack = ?
            """, [racetrack]).fetchone()
        return {
            "track_temp_c": float(row[0]) if row[0] is not None else 20.0,
            "air_temp_c": float(row[1]) if row[1] is not None else 18.0,
            "humidity_pct": float(row[2]) if row[2] is not None else 60.0,
            "weather": row[3] if row[3] is not None else "晴",
            "track_condition": row[4] if row[4] is not None else "良走路",
        }
    finally:
        con.close()


def get_races_columns() -> list[str]:
    """races テーブルの列名一覧。"""
    con = get_connection()
    try:
        return con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'races' "
            "ORDER BY ordinal_position"
        ).df()["column_name"].tolist()
    finally:
        con.close()


def upsert_race_rows(rows: list[dict]) -> dict:
    """races テーブルへ行をupsertする。

    キー: (yyyymmdd, pid, race_no, bike_no)
    既存キーの行は DELETE → INSERT で置き換え。

    Returns
    -------
    dict { 'inserted': N, 'replaced': N, 'races_updated': set[(ymd,pid,rno)] }
    """
    init_database(force=False)
    if not rows:
        return {"inserted": 0, "replaced": 0, "races_updated": []}

    df = pd.DataFrame(rows)
    # 型を整える
    df["yyyymmdd"] = pd.to_numeric(df["yyyymmdd"], errors="coerce").astype("Int64")
    df["pid"] = pd.to_numeric(df["pid"], errors="coerce").astype("Int64")
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").astype("Int64")
    df["bike_no"] = pd.to_numeric(df["bike_no"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["yyyymmdd", "pid", "race_no", "bike_no"])

    cols = get_races_columns()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].copy()

    con = get_connection()
    try:
        race_keys = (
            df[["yyyymmdd", "pid", "race_no"]]
            .drop_duplicates().values.tolist()
        )
        replaced = 0
        for ymd, pid, rno in race_keys:
            n = con.execute("""
                SELECT count(*) FROM races
                WHERE yyyymmdd = ? AND pid = ? AND race_no = ?
            """, [int(ymd), int(pid), int(rno)]).fetchone()[0]
            replaced += n
            if n > 0:
                con.execute("""
                    DELETE FROM races
                    WHERE yyyymmdd = ? AND pid = ? AND race_no = ?
                """, [int(ymd), int(pid), int(rno)])

        con.register("incoming_df", df)
        con.execute("INSERT INTO races SELECT * FROM incoming_df")
        con.unregister("incoming_df")

        # finish_position を再計算 (該当レースのみ)
        for ymd, pid, rno in race_keys:
            con.execute("""
                UPDATE races SET finish_position = CASE
                    WHEN bike_no = finish_1st THEN 1
                    WHEN bike_no = finish_2nd THEN 2
                    WHEN bike_no = finish_3rd THEN 3
                    ELSE NULL
                END
                WHERE yyyymmdd = ? AND pid = ? AND race_no = ?
            """, [int(ymd), int(pid), int(rno)])

        return {
            "inserted": len(df),
            "replaced": replaced,
            "races_updated": race_keys,
        }
    finally:
        con.close()


def get_imported_race_dates() -> dict[int, set[int]]:
    """各 pid について、races テーブルに存在する yyyymmdd の集合を返す。

    自動検出 (未取り込みレース判定) で使用する。
    """
    init_database(force=False)
    con = get_connection()
    try:
        rows = con.execute(
            "SELECT DISTINCT pid, yyyymmdd FROM races "
            "WHERE pid IS NOT NULL AND yyyymmdd IS NOT NULL"
        ).fetchall()
        out: dict[int, set[int]] = {}
        for pid, ymd in rows:
            out.setdefault(int(pid), set()).add(int(ymd))
        return out
    finally:
        con.close()


def get_latest_imported_date() -> int | None:
    """races テーブルの最大 yyyymmdd (整数) を返す。空なら None。"""
    init_database(force=False)
    con = get_connection()
    try:
        row = con.execute("SELECT MAX(yyyymmdd) FROM races").fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])
    finally:
        con.close()


def load_rows_for_keys(race_keys: list[tuple]) -> pd.DataFrame:
    """指定レースキー (yyyymmdd, pid, race_no) の学習用 DataFrame を返す。"""
    if not race_keys:
        return pd.DataFrame()
    init_database(force=False)
    con = get_connection()
    try:
        clauses = " OR ".join(
            ["(yyyymmdd=? AND pid=? AND race_no=?)" for _ in race_keys])
        params: list = []
        for ymd, pid, rno in race_keys:
            params.extend([int(ymd), int(pid), int(rno)])
        df = con.execute(f"""
            SELECT
                r.*,
                pf.s_power, pf.solo_power, pf.chase_power,
                pf.rain_skill, pf.course_pref
            FROM races r
            LEFT JOIN player_features pf
              ON norm_name(r.player_name) = pf.player_name
            WHERE ({clauses}) AND r.bike_no IS NOT NULL
        """, params).df()
        return df
    finally:
        con.close()


if __name__ == "__main__":
    init_database(force=False)
    print("[ok] database initialized:", DB_PATH)
    con = get_connection()
    try:
        n_races = con.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        n_players = con.execute(
            "SELECT COUNT(*) FROM player_features").fetchone()[0]
        print(f"  races rows         : {n_races:,}")
        print(f"  player_features    : {n_players:,}")
    finally:
        con.close()
