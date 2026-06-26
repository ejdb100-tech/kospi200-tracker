#!/usr/bin/env python3
"""
코스피200 50일 이격도 + SK하이닉스/삼성전자 MDD 트래커 데이터 생성기.
출력: data.json (정적 사이트가 fetch로 읽음)

데이터 소스: FinanceDataReader (네이버 기반).
  * 원래 pykrx를 썼으나 KRX 보안정책 변경으로 지수 조회가 깨져(KeyError: '지수명')
    FinanceDataReader로 교체함. 코스피200 = 'KS200', 종목 = 종목코드.

지표
- 이격도 = 코스피200 종가 / 50일 이동평균 * 100
- 낙폭/MDD는 프론트(index.html)에서 기간별로 계산
"""
import json
import datetime
import sys

import pandas as pd
import FinanceDataReader as fdr

KST = datetime.timezone(datetime.timedelta(hours=9))
KOSPI200_SYMBOL = "KS200"
STOCKS = {"000660": "SK하이닉스", "005930": "삼성전자"}
LOOKBACK_DAYS = 365 * 5 + 120   # 5년 + 50일 이동평균 워밍업 여유


def fetch_close(symbol, start_iso):
    df = fdr.DataReader(symbol, start_iso)
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError(f"{symbol} 데이터를 받지 못했습니다.")
    s = df["Close"].astype(float).dropna()
    s.index = pd.to_datetime(s.index)
    return s


def main():
    today = datetime.datetime.now(KST).date()
    start_iso = (today - datetime.timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    cutoff = today - datetime.timedelta(days=365 * 5)

    # --- 코스피200: 50일 이격도 ---
    k200 = fetch_close(KOSPI200_SYMBOL, start_iso)
    ma50 = k200.rolling(50).mean()
    disp = k200 / ma50 * 100
    valid = ma50.notna()
    k200, ma50, disp = k200[valid], ma50[valid], disp[valid]
    keep = k200.index.date >= cutoff
    k200, ma50, disp = k200[keep], ma50[keep], disp[keep]

    kospi200 = {
        "dates": [d.strftime("%Y-%m-%d") for d in k200.index],
        "close": [round(float(x), 2) for x in k200.values],
        "ma50": [round(float(x), 2) for x in ma50.values],
        "disparity": [round(float(x), 2) for x in disp.values],
    }

    # --- 종목별 종가 ---
    stocks = {}
    for code, name in STOCKS.items():
        s = fetch_close(code, start_iso)
        s = s[s.index.date >= cutoff]
        stocks[code] = {
            "name": name,
            "dates": [d.strftime("%Y-%m-%d") for d in s.index],
            "close": [round(float(x), 2) for x in s.values],
        }

    out = {
        "updated_at": datetime.datetime.now(KST).isoformat(timespec="seconds"),
        "is_sample": False,
        "kospi200": kospi200,
        "stocks": stocks,
    }
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    print(f"updated {out['updated_at']} | KOSPI200 disparity = {kospi200['disparity'][-1]}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
