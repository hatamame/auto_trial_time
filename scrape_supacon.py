"""
Gamboo「スパコン直前予想」スクレイパー

autorace_schedule.csv に記載された全レースについて、Gamboo の
スパコン直前予想ページ (pt=8) を取得し、機械学習で扱いやすい
long-format CSV (車1台 × 1レース = 1行) を生成する。

スクレイピング対象:
  - 天候 / 走路状況 / 走路温度 / 気温 / 湿度
  - 選手ごと: 車番・ランク・期別・選手名・車級・車名・ハンデ・
              試走タイム・予想競走タイム・予想着順
  - レース結果: 1着 / 2着 / 3着 / 2連単払戻金 / 人気

依存:  pip install requests beautifulsoup4 lxml pandas tqdm

URL 例:
  https://gamboo.jp/autorace/program?pid=2&rdt=20200102&rno=1&pt=8
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ----------------------------------------------------------------------
# 定数
# ----------------------------------------------------------------------

PID_MAP: dict[str, int] = {
    "川口": 2,
    "川口2": 2,   # 同じ会場(夜開催) — pid は同一
    "伊勢崎": 3,
    "浜松": 4,
    "飯塚": 5,
    "山陽": 6,
}

URL_TEMPLATE = "https://gamboo.jp/autorace/program?pid={pid}&rdt={rdt}&rno={rno}&pt=8"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
}

OUTPUT_COLUMNS = [
    "date", "yyyymmdd", "racetrack", "pid", "race_no",
    "grade", "event_type", "event_name",
    "weather", "track_condition", "track_temp_c", "air_temp_c", "humidity_pct",
    "bike_no", "rank", "player_lg", "license_period",
    "player_name", "bike_class", "bike_name", "handicap_m",
    "trial_time", "expected_race_time", "expected_order", "focus_mark",
    "finish_1st", "finish_2nd", "finish_3rd",
    "exacta_combo", "exacta_payout_yen", "exacta_popularity",
    "canceled", "scrape_status", "url",
]


# ----------------------------------------------------------------------
# データクラス
# ----------------------------------------------------------------------

@dataclass
class RaceMeta:
    date: str
    yyyymmdd: str
    racetrack: str
    pid: int
    race_no: int
    grade: str = ""
    event_type: str = ""
    event_name: str = ""


@dataclass
class WeatherInfo:
    weather: str = ""
    track_condition: str = ""
    track_temp_c: float | None = None
    air_temp_c: float | None = None
    humidity_pct: float | None = None


@dataclass
class RaceResult:
    finish_1st: int | None = None
    finish_2nd: int | None = None
    finish_3rd: int | None = None
    exacta_combo: str = ""
    exacta_payout_yen: int | None = None
    exacta_popularity: int | None = None


@dataclass
class PlayerRow:
    bike_no: int
    rank: str = ""
    player_lg: str = ""
    license_period: str = ""
    player_name: str = ""
    bike_class: str = ""
    bike_name: str = ""
    handicap_m: float | None = None
    trial_time: float | None = None
    expected_race_time: float | None = None
    expected_order: int | None = None
    focus_mark: str = ""


# ----------------------------------------------------------------------
# パース処理
# ----------------------------------------------------------------------

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# expect['w']['t']['bike'] = {'order':'8','bike_no':'1','race_time':'3.539...',
#                             'traial_time':'3.48','focus':{'src':'...','alt':'...'}};
EXPECT_LINE_RE = re.compile(
    r"expect\[\s*'([^']+)'\s*\]\[\s*'([^']+)'\s*\]\[\s*'([^']+)'\s*\]\s*=\s*"
    r"\{([^}]*\{[^}]*\}[^}]*|[^}]*)\};",
    re.DOTALL,
)
EXPECT_KV_RE = re.compile(r"'([^']+)'\s*:\s*'([^']*)'")
EXPECT_FOCUS_RE = re.compile(
    r"'focus'\s*:\s*\{\s*'src'\s*:\s*'([^']*)'\s*,\s*'alt'\s*:\s*'([^']*)'\s*\}"
)


def _decode_js_string(s: str) -> str:
    """JS の \\uXXXX エスケープを Python 文字列に戻す。"""
    try:
        return s.encode("utf-8").decode("unicode_escape")
    except Exception:
        return s


def _parse_expect_js(html: str) -> dict[tuple[str, str, str], dict]:
    """`expect[..][..][..] = {...};` 行を辞書化して返す。
    key = (weather_idx, temp_idx, bike_no) (全て str)
    """
    out: dict[tuple[str, str, str], dict] = {}
    # 巨大な HTML 全体に regex を当てると遅いので冒頭の <script> ブロックだけ
    head = html[: min(len(html), 200_000)]
    for m in EXPECT_LINE_RE.finditer(head):
        w = _decode_js_string(m.group(1))
        t = _decode_js_string(m.group(2))
        bike = _decode_js_string(m.group(3))
        body = m.group(4)
        # 数字以外のキー (confidence, vote_recommend, valid 等) はスキップ
        if not bike.isdigit():
            continue
        data: dict[str, str] = {}
        for k, v in EXPECT_KV_RE.findall(body):
            data[k] = _decode_js_string(v)
        fm = EXPECT_FOCUS_RE.search(body)
        if fm:
            data["focus_alt"] = _decode_js_string(fm.group(2))
        out[(w, t, bike)] = data
    return out


def _select_expect_slice(
    expect: dict[tuple[str, str, str], dict],
    weather: str,
    track_temp_c: float | None,
) -> dict[int, dict]:
    """天候 + 走路温度から該当する `expect[w][t]` を選び bike_no → data に変換。"""
    # weather: 晴/曇/雪 等は 0、雨を含むものは 1
    w = "1" if weather and "雨" in weather else "0"
    # temperature bucket
    if w == "1":
        t = "0"
    elif track_temp_c is None:
        t = "0"
    elif track_temp_c < 20:
        t = "1"
    elif track_temp_c < 35:
        t = "2"
    elif track_temp_c < 50:
        t = "3"
    else:
        t = "4"
    out: dict[int, dict] = {}
    for (ww, tt, bike), data in expect.items():
        if ww == w and tt == t:
            try:
                out[int(bike)] = data
            except ValueError:
                pass
    # フォールバック: 完全一致が無ければ温度 0(全) を使う
    if not out and t != "0":
        for (ww, tt, bike), data in expect.items():
            if ww == w and tt == "0":
                try:
                    out[int(bike)] = data
                except ValueError:
                    pass
    return out


def _to_float(text: str | None) -> float | None:
    if not text:
        return None
    m = NUM_RE.search(text)
    return float(m.group()) if m else None


def _to_int(text: str | None) -> int | None:
    if not text:
        return None
    m = NUM_RE.search(text)
    return int(float(m.group())) if m else None


def parse_weather(soup: BeautifulSoup) -> WeatherInfo:
    """`<ul class="grayrace">` から天候情報を抽出。"""
    info = WeatherInfo()
    for ul in soup.select("ul.grayrace"):
        items = ul.find_all("li")
        if len(items) < 2:
            continue
        label = items[0].get_text(strip=True)
        value = items[1].get_text(strip=True)
        if label == "天候":
            info.weather = value
        elif label == "走路状況":
            info.track_condition = value
        elif label == "走路温度":
            info.track_temp_c = _to_float(value)
        elif label == "気温":
            info.air_temp_c = _to_float(value)
        elif label == "湿度":
            info.humidity_pct = _to_float(value)
    return info


def parse_result(soup: BeautifulSoup) -> RaceResult:
    """ページ上部「レース結果」セクションから 1〜3着・払戻金を抽出。"""
    result = RaceResult()
    block = soup.find("div", id="charttitle03")
    if block is None:
        return result

    # 着順 (img alt="1" 〜 "8") を順に拾う
    finish_lis = block.find("li", class_="left")
    if finish_lis is not None:
        order_imgs = finish_lis.select("ul li img")
        bike_numbers: list[int] = []
        for img in order_imgs:
            alt = img.get("alt", "").strip()
            n = _to_int(alt)
            if n is not None:
                bike_numbers.append(n)
        if len(bike_numbers) >= 1:
            result.finish_1st = bike_numbers[0]
        if len(bike_numbers) >= 2:
            result.finish_2nd = bike_numbers[1]
        if len(bike_numbers) >= 3:
            result.finish_3rd = bike_numbers[2]

    # 2連単・払戻金
    exacta_blocks = block.find_all("li", class_="left")
    if len(exacta_blocks) >= 2:
        ex = exacta_blocks[1]
        nums: list[int] = []
        for img in ex.select("img"):
            n = _to_int(img.get("alt", ""))
            if n is not None:
                nums.append(n)
        if len(nums) >= 2:
            result.exacta_combo = f"{nums[0]}-{nums[1]}"
        # 払戻金と人気は "720円　4番人気" の様な文字列
        text = ex.get_text(" ", strip=True)
        m_pay = re.search(r"([\d,]+)\s*円", text)
        if m_pay:
            result.exacta_payout_yen = int(m_pay.group(1).replace(",", ""))
        m_pop = re.search(r"(\d+)\s*番人気", text)
        if m_pop:
            result.exacta_popularity = int(m_pop.group(1))
    return result


def parse_players(soup: BeautifulSoup) -> list[PlayerRow]:
    """`#program1 > #chart > table.chartA` から各選手データを抽出。"""
    rows: list[PlayerRow] = []
    program = soup.find("div", id="program1")
    if program is None:
        return rows

    chart = program.find("div", id="chart")
    if chart is None:
        return rows

    table = chart.find("table", class_="chartA")
    if table is None:
        return rows

    # ヘッダ行 2行を除いた tr が選手データ
    trs = table.find_all("tr", recursive=False)
    # bs4 では <tbody> 内の tr を返すため recursive=False は効かないので
    # 別途フィルタする
    if not trs:
        trs = table.find_all("tr")

    bike_no = 0
    for tr in trs:
        player_td = tr.find("td", class_="player")
        if player_td is None:
            continue
        bike_no += 1

        row = PlayerRow(bike_no=bike_no)

        rank_ul = player_td.find("ul", class_="rank")
        if rank_ul is not None:
            lis = rank_ul.find_all("li")
            if len(lis) >= 1:
                row.rank = lis[0].get_text(strip=True)
            if len(lis) >= 2:
                img = lis[1].find("img")
                lg = img.get("alt", "").strip() if img else ""
                # alt は "川　口" 等の全角スペース入り — 取り除く
                row.player_lg = re.sub(r"\s|　", "", lg)
            if len(lis) >= 3:
                row.license_period = lis[2].get_text(strip=True).replace("期", "")

        name_div = player_td.find("div", class_="name")
        if name_div is not None:
            a = name_div.find("a")
            row.player_name = (a.get_text(strip=True) if a else
                               name_div.get_text(strip=True))
            row.player_name = re.sub(r"\s+", "", row.player_name)

        bike_div = player_td.find("div", class_="bikename")
        if bike_div is not None:
            cls_img = bike_div.find("img")
            if cls_img is not None:
                row.bike_class = cls_img.get("alt", "").strip()
            # img を取り除いた残りが車名
            text = bike_div.get_text(" ", strip=True)
            text = re.sub(r"^\s*車級\s*", "", text)
            text = re.sub(r"^\s*\d+\s*", "", text)
            row.bike_name = text.strip()

        # ハンデは td > div.handicap
        handi_div = tr.find("div", class_="handicap")
        if handi_div is not None:
            row.handicap_m = _to_float(handi_div.get_text(strip=True))

        # 印 / 予想着順 は td id を頼りに探す (数字 or img alt のどちらか)
        focus_td = tr.find("td", id=re.compile(r"^expect-focus-\d+$"))
        if focus_td is not None:
            img = focus_td.find("img")
            row.focus_mark = (img.get("alt", "").strip() if img
                              else focus_td.get_text(strip=True))

        order_td = tr.find("td", id=re.compile(r"^expect-order-\d+$"))
        if order_td is not None:
            img = order_td.find("img")
            text = (img.get("alt", "").strip() if img
                    else order_td.get_text(strip=True))
            row.expected_order = _to_int(text)

        racetime_td = tr.find("td", id=re.compile(r"^expect-racetime-\d+$"))
        if racetime_td is not None:
            row.expected_race_time = _to_float(racetime_td.get_text(strip=True))

        # 試走タイム / 予想T のうち、id が無いセルが試走T。
        # focus / order / racetime / player / handicap を除いた残りの
        # 直下 td から数値を持つもののうち、racetime より前にある td が試走T。
        all_tds = tr.find_all("td", recursive=False)
        skip_ids = {focus_td and focus_td.get("id"),
                    order_td and order_td.get("id"),
                    racetime_td and racetime_td.get("id")}
        for td in all_tds:
            if td is player_td:
                continue
            tid = td.get("id")
            if tid in skip_ids:
                continue
            if td.find("div", class_="handicap") is not None:
                continue
            # ネストの table.chartB(直近4走の成績) はスキップ
            if td.find("table", class_="chartB") is not None:
                continue
            txt = td.get_text(strip=True)
            val = _to_float(txt)
            if val is not None and row.trial_time is None:
                row.trial_time = val
                break

        rows.append(row)

    return rows


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------

class GambooFetcher:
    def __init__(self, delay: float = 1.0, timeout: float = 20.0,
                 retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self._last_request_ts: float = 0.0

    def fetch(self, url: str) -> tuple[str | None, int]:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        last_status = 0
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                self._last_request_ts = time.time()
                last_status = resp.status_code
                if resp.status_code == 200:
                    resp.encoding = resp.apparent_encoding or "utf-8"
                    return resp.text, 200
                if resp.status_code == 404:
                    return None, 404
            except requests.RequestException as e:
                last_status = -1
                # ネットワーク系エラー — リトライ
                time.sleep(min(2 ** attempt, 10))
                continue
            time.sleep(min(2 ** attempt, 10))
        return None, last_status


# ----------------------------------------------------------------------
# レース単位処理
# ----------------------------------------------------------------------

def race_to_rows(meta: RaceMeta, html: str) -> list[dict]:
    """1レース分の HTML を選手×レースの行リストへ変換。"""
    soup = BeautifulSoup(html, "lxml")

    weather = parse_weather(soup)
    result = parse_result(soup)
    players = parse_players(soup)

    # ライブページでは予想値 (expected_*, focus_mark) が JS から注入される。
    # HTML 直読みで取れていない場合は JS の expect 辞書から補完する。
    need_js = any(p.expected_order is None or p.expected_race_time is None
                  or not p.focus_mark for p in players)
    if need_js and players:
        expect_all = _parse_expect_js(html)
        slice_ = _select_expect_slice(expect_all, weather.weather,
                                       weather.track_temp_c)
        for p in players:
            data = slice_.get(p.bike_no)
            if not data:
                continue
            if p.expected_order is None:
                p.expected_order = _to_int(data.get("order"))
            if p.expected_race_time is None:
                p.expected_race_time = _to_float(data.get("race_time"))
            if p.trial_time is None:
                p.trial_time = _to_float(data.get("traial_time"))
            if not p.focus_mark:
                p.focus_mark = data.get("focus_alt", "")

    # 出走表が空 = レースが存在しない or 中止扱い
    if not players:
        return [{
            **asdict(meta),
            **asdict(weather),
            **asdict(result),
            "bike_no": None, "rank": "", "player_lg": "",
            "license_period": "", "player_name": "",
            "bike_class": "", "bike_name": "", "handicap_m": None,
            "trial_time": None, "expected_race_time": None,
            "expected_order": None,
            "canceled": 1,
            "scrape_status": "no_players",
            "url": URL_TEMPLATE.format(pid=meta.pid, rdt=meta.yyyymmdd,
                                      rno=meta.race_no),
        }]

    out: list[dict] = []
    for p in players:
        row = {
            **asdict(meta),
            **asdict(weather),
            **asdict(p),
            **asdict(result),
            "canceled": 0,
            "scrape_status": "ok",
            "url": URL_TEMPLATE.format(pid=meta.pid, rdt=meta.yyyymmdd,
                                      rno=meta.race_no),
        }
        out.append(row)
    return out


# ----------------------------------------------------------------------
# スケジュール展開
# ----------------------------------------------------------------------

def load_schedule(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).copy()
    df["yyyymmdd"] = df["date_dt"].dt.strftime("%Y%m%d")
    df["date_iso"] = df["date_dt"].dt.strftime("%Y-%m-%d")
    return df


def iter_races(schedule: pd.DataFrame,
               start: date | None,
               end: date | None,
               tracks: set[str] | None,
               max_rno: int) -> Iterator[RaceMeta]:
    for _, row in schedule.iterrows():
        d: pd.Timestamp = row["date_dt"]
        if start and d.date() < start:
            continue
        if end and d.date() > end:
            continue
        track = row["racetrack"]
        if tracks and track not in tracks:
            continue
        pid = PID_MAP.get(track)
        if pid is None:
            continue
        for rno in range(1, max_rno + 1):
            yield RaceMeta(
                date=row["date_iso"],
                yyyymmdd=row["yyyymmdd"],
                racetrack=track,
                pid=pid,
                race_no=rno,
                grade=row.get("grade", ""),
                event_type=row.get("event_type", ""),
                event_name=row.get("event_name", ""),
            )


# ----------------------------------------------------------------------
# CSV 出力(レジューム対応)
# ----------------------------------------------------------------------

class CsvWriter:
    """append-mode で書き出すラッパ。既存ファイル末尾に続ける。"""

    def __init__(self, path: Path):
        self.path = path
        is_new = not path.exists() or path.stat().st_size == 0
        self.fp = open(path, "a", encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.fp, fieldnames=OUTPUT_COLUMNS,
                                     extrasaction="ignore")
        if is_new:
            self.writer.writeheader()
            self.fp.flush()

    def write(self, rows: Iterable[dict]):
        for row in rows:
            self.writer.writerow(row)
        self.fp.flush()

    def close(self):
        self.fp.close()


def load_done_keys(path: Path) -> set[tuple[str, int, int]]:
    """既出力 CSV から (yyyymmdd, pid, race_no) の集合を作る。"""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: set[tuple[str, int, int]] = set()
    # 大きくなり得るので chunk 読込
    for chunk in pd.read_csv(path, usecols=["yyyymmdd", "pid", "race_no"],
                             dtype={"yyyymmdd": str, "pid": "Int64",
                                    "race_no": "Int64"},
                             chunksize=100_000):
        chunk = chunk.dropna()
        for ymd, pid, rno in chunk.itertuples(index=False, name=None):
            done.add((str(ymd), int(pid), int(rno)))
    return done


# ----------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--schedule", type=Path,
                   default=Path(__file__).with_name("autorace_schedule.csv"),
                   help="開催日程 CSV")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).with_name("supacon_dataset.csv"),
                   help="出力 CSV (追記モードで開く)")
    p.add_argument("--start", type=str, default=None,
                   help="開始日 (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None,
                   help="終了日 (YYYY-MM-DD). 既定: 昨日まで")
    p.add_argument("--tracks", type=str, default=None,
                   help="場フィルタ (カンマ区切り、例: 川口,伊勢崎)")
    p.add_argument("--max-rno", type=int, default=12,
                   help="1日あたり最大レース番号 (既定: 12)")
    p.add_argument("--delay", type=float, default=1.2,
                   help="リクエスト間隔(秒)")
    p.add_argument("--retries", type=int, default=3,
                   help="HTTP リトライ回数")
    p.add_argument("--timeout", type=float, default=20.0,
                   help="HTTP タイムアウト(秒)")
    p.add_argument("--no-resume", action="store_true",
                   help="既存出力を参照せず全レースを再取得")
    p.add_argument("--limit", type=int, default=0,
                   help="デバッグ用: 取得するレース数の上限 (0=制限なし)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.schedule.exists():
        print(f"[error] schedule not found: {args.schedule}", file=sys.stderr)
        return 2

    start = (datetime.strptime(args.start, "%Y-%m-%d").date()
             if args.start else None)
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        # 未来のレースは結果が出ていないので今日より前を既定とする
        end = date.today()

    tracks = (set(t.strip() for t in args.tracks.split(",") if t.strip())
              if args.tracks else None)

    schedule = load_schedule(args.schedule)
    races = list(iter_races(schedule, start, end, tracks, args.max_rno))
    print(f"[info] target races (before resume filter): {len(races):,}")

    done_keys: set[tuple[str, int, int]] = set()
    if not args.no_resume:
        done_keys = load_done_keys(args.output)
        if done_keys:
            print(f"[info] resume mode: {len(done_keys):,} races already done")
            races = [r for r in races
                     if (r.yyyymmdd, r.pid, r.race_no) not in done_keys]
            print(f"[info] remaining races: {len(races):,}")

    if args.limit:
        races = races[:args.limit]

    if not races:
        print("[info] nothing to do.")
        return 0

    fetcher = GambooFetcher(delay=args.delay, timeout=args.timeout,
                            retries=args.retries)
    writer = CsvWriter(args.output)

    ok = miss = err = 0
    try:
        bar = tqdm(races, desc="scrape", unit="race")
        for meta in bar:
            url = URL_TEMPLATE.format(pid=meta.pid, rdt=meta.yyyymmdd,
                                      rno=meta.race_no)
            html, status = fetcher.fetch(url)
            if html is None:
                # 404 は欠番として 1行だけ残す
                writer.write([{
                    **asdict(meta),
                    "weather": "", "track_condition": "",
                    "track_temp_c": None, "air_temp_c": None,
                    "humidity_pct": None,
                    "bike_no": None,
                    "canceled": 1,
                    "scrape_status": f"http_{status}",
                    "url": url,
                }])
                if status == 404:
                    miss += 1
                else:
                    err += 1
                continue

            try:
                rows = race_to_rows(meta, html)
                writer.write(rows)
                if rows and rows[0].get("scrape_status") == "ok":
                    ok += 1
                else:
                    miss += 1
            except Exception as e:
                err += 1
                writer.write([{
                    **asdict(meta),
                    "canceled": 1,
                    "scrape_status": f"parse_error:{type(e).__name__}",
                    "url": url,
                }])
            bar.set_postfix(ok=ok, miss=miss, err=err)
    finally:
        writer.close()

    print(f"[done] ok={ok}, missing={miss}, errors={err}")
    print(f"[done] output -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
