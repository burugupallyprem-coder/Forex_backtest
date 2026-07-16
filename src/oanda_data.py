"""OANDA v3 candle data (mid prices). Practice host by default."""

import os
import time

import pandas as pd
import requests


def _base():
    env = os.environ.get("OANDA_ENV", "practice")
    return ("https://api-fxtrade.oanda.com" if env == "live"
            else "https://api-fxpractice.oanda.com")


def fetch_candles(instrument, start, end, granularity="M15", count=5000):
    """Returns [ts(UTC), open, high, low, close, volume] for complete candles."""
    headers = {"Authorization": f"Bearer {os.environ['OANDA_API_KEY']}"}
    url = f"{_base()}/v3/instruments/{instrument}/candles"
    rows = []
    frm = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    session = requests.Session()
    for _ in range(2000):                      # hard page cap
        params = {"granularity": granularity, "price": "M", "count": count,
                  "from": frm.strftime("%Y-%m-%dT%H:%M:%SZ")}
        resp = session.get(url, params=params, headers=headers, timeout=60)
        if resp.status_code == 429:
            time.sleep(2)
            continue
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        if not candles:
            break
        for c in candles:
            if not c.get("complete"):
                continue
            t = pd.Timestamp(c["time"])
            if t >= end_ts:
                break
            m = c["mid"]
            rows.append((t, float(m["o"]), float(m["h"]), float(m["l"]),
                         float(m["c"]), int(c.get("volume", 0))))
        last = pd.Timestamp(candles[-1]["time"])
        if last >= end_ts or last <= frm:
            break
        frm = last + pd.Timedelta(seconds=1)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
