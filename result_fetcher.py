"""完走したレースの結果を取得するモジュール。

公開ページ `/autorace/program?pt=8` を取得し、scrape_supacon.py の
パース処理を使って races テーブルに upsert 可能な行リストを返す。
"""
from __future__ import annotations

from datetime import datetime

from bs4 import BeautifulSoup
import requests

from scrape_supacon import (
    RaceMeta, race_to_rows, URL_TEMPLATE, PID_MAP, DEFAULT_HEADERS,
)

TRACK_FROM_PID = {v: k for k, v in PID_MAP.items() if not k.endswith("2")}


def fetch_race_result(pid: int, yyyymmdd: str, race_no: int,
                      grade: str = "", event_type: str = "",
                      event_name: str = "",
                      timeout: float = 20.0) -> dict:
    """1 レース分の結果データを取得する。

    Returns
    -------
    dict
        {
          'rows': list[dict],       # races テーブル互換の行
          'has_result': bool,       # 1〜3着が全て埋まっているか
          'url': str,
          'status': int,
        }
    """
    url = URL_TEMPLATE.format(pid=pid, rdt=yyyymmdd, rno=race_no)
    racetrack = TRACK_FROM_PID.get(pid, "")
    date_iso = ""
    try:
        date_iso = datetime.strptime(yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        pass

    meta = RaceMeta(
        date=date_iso,
        yyyymmdd=str(yyyymmdd),
        racetrack=racetrack,
        pid=pid,
        race_no=int(race_no),
        grade=grade,
        event_type=event_type,
        event_name=event_name,
    )

    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    if resp.status_code != 200:
        return {"rows": [], "has_result": False,
                "url": url, "status": resp.status_code}
    resp.encoding = resp.apparent_encoding or "utf-8"

    rows = race_to_rows(meta, resp.text)
    has_result = bool(rows) and all(
        r.get("finish_1st") and r.get("finish_2nd") and r.get("finish_3rd")
        for r in rows[:1]
    )
    return {"rows": rows, "has_result": has_result,
            "url": url, "status": 200}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--rdt", type=str, required=True)
    ap.add_argument("--rno", type=int, required=True)
    args = ap.parse_args()

    result = fetch_race_result(args.pid, args.rdt, args.rno)
    print("status:", result["status"], "has_result:", result["has_result"])
    print("n_rows:", len(result["rows"]))
    if result["rows"]:
        # 結果系のみ表示
        r0 = result["rows"][0]
        print("finish:", r0.get("finish_1st"), r0.get("finish_2nd"),
              r0.get("finish_3rd"))
        print("payout:", r0.get("exacta_payout_yen"))
        print("weather:", r0.get("weather"),
              r0.get("track_condition"), r0.get("track_temp_c"))
        print(json.dumps(r0, ensure_ascii=False)[:300])
