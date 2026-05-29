"""Gamboo の出走表ページから出走情報を取得する。

主 URL: https://gamboo.jp/yoso/autorace/?pid={pid}&rdt={yyyymmdd}&rno={race_no}
  (要ログインのため出走表 HTML が返らない場合がある)

フォールバック URL (公開ページ):
  https://gamboo.jp/autorace/program?pid={pid}&rdt={yyyymmdd}&rno={race_no}&pt=8
"""
from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup

# 既存スクレイパの天候/選手パース処理を流用
from scrape_supacon import parse_players, parse_weather, PID_MAP

URL_TEMPLATE = ("https://gamboo.jp/yoso/autorace/"
                "?pid={pid}&rdt={yyyymmdd}&rno={race_no}")
PROGRAM_URL = ("https://gamboo.jp/autorace/program"
               "?pid={pid}&rdt={yyyymmdd}&rno={race_no}&pt=8")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
}


def build_url(pid: int, yyyymmdd: str, race_no: int) -> str:
    return URL_TEMPLATE.format(pid=pid, yyyymmdd=yyyymmdd, race_no=race_no)


def _get(url: str, timeout: float) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return BeautifulSoup(resp.text, "lxml")


def fetch_entry(pid: int, yyyymmdd: str, race_no: int,
                timeout: float = 20.0) -> dict:
    """1 レースの出走情報を取得する。

    指定の /yoso/ ページから取れない場合、公開ページ
    /autorace/program?pt=8 にフォールバックする。
    """
    url = build_url(pid, yyyymmdd, race_no)
    soup = _get(url, timeout)
    w = parse_weather(soup)
    players_raw = parse_players(soup)
    source_url = url
    if not players_raw:
        fallback = PROGRAM_URL.format(
            pid=pid, yyyymmdd=yyyymmdd, race_no=race_no)
        try:
            soup_fb = _get(fallback, timeout)
            players_raw_fb = parse_players(soup_fb)
            if players_raw_fb:
                players_raw = players_raw_fb
                soup = soup_fb
                w = parse_weather(soup)
                source_url = fallback
        except requests.RequestException:
            pass
    players: list[dict] = []
    for p in players_raw:
        players.append({
            "bike_no": p.bike_no,
            "rank": p.rank,
            "player_lg": p.player_lg,
            "license_period": p.license_period,
            "player_name": p.player_name,
            "bike_class": p.bike_class,
            "bike_name": p.bike_name,
            "handicap_m": p.handicap_m,
            "trial_time_actual": p.trial_time,  # サイト掲載の試走T (あれば)
        })

    # 会場名 (タイトル等から推定)
    racetrack = ""
    for name, p in PID_MAP.items():
        if p == pid:
            racetrack = name
            break
    title_tag = soup.find("title")
    if title_tag is not None:
        m = re.search(
            r"(川口|伊勢崎|浜松|飯塚|山陽)", title_tag.get_text())
        if m:
            racetrack = m.group(1)

    return {
        "url": url,
        "source_url": source_url,
        "racetrack": racetrack,
        "pid": pid,
        "yyyymmdd": yyyymmdd,
        "race_no": race_no,
        "weather": w.weather or None,
        "track_condition": w.track_condition or None,
        "track_temp_c": w.track_temp_c,
        "air_temp_c": w.air_temp_c,
        "humidity_pct": w.humidity_pct,
        "players": players,
    }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--rdt", type=str, required=True, help="YYYYMMDD")
    p.add_argument("--rno", type=int, required=True)
    args = p.parse_args()

    data = fetch_entry(args.pid, args.rdt, args.rno)
    print(json.dumps(data, ensure_ascii=False, indent=2))
