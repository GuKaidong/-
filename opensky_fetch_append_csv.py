#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每天抓取 OpenSky /states/all（只能近 1 小时历史），将结果追加写入当天目录 CSV（自动去重）。
不使用环境变量：密钥必须通过命令行参数传入，或通过 --secrets_json 读取本地 JSON。

输出：
  data/opensky_daily/YYYYMMDD/opensky_raw_states.csv.gz
  data/opensky_daily/YYYYMMDD/opensky_lowalt_states.csv.gz
  data/opensky_daily/YYYYMMDD/fetch_summary.json

用法 1：命令行直接传入（最简单）
python -u 1217/opensky_fetch_append_csv.py ^
  --client_id "gkd19850773346@outlook.com-api-client" --client_secret "tCyapXumMBr9n9qoiqvqKm6N90UpGCnT" ^
  --out_root data/opensky_daily ^
  --duration_minutes 55 --time_step 10 ^
  --lat_min 22.5 --lat_max 24.0 --lon_min 112.5 --lon_max 114.5 ^
  --max_altitude_m 3000 --sleep_per_call 0.05

用法 2：从本地 secrets.json 读取（不走环境变量）
python -u src/data_prep/opensky_fetch_append_csv.py ^
  --secrets_json secrets/opensky_secrets.json ^
  --out_root data/opensky_daily ^
  --duration_minutes 40 --time_step 10 ^
  --lat_min 22.5 --lat_max 24.0 --lon_min 112.5 --lon_max 114.5
secrets.json 格式：
{
  "client_id": "xxx",
  "client_secret": "yyy"
}
"""

import os
import time
import json
import argparse
from datetime import datetime, timezone

import pandas as pd
import requests
from tqdm import tqdm


TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
STATES_URL = "https://opensky-network.org/api/states/all"


def load_secrets(secrets_json: str):
    with open(secrets_json, "r", encoding="utf-8") as f:
        j = json.load(f)
    cid = j.get("client_id", None)
    csec = j.get("client_secret", None)
    return cid, csec


def obtain_token(client_id: str, client_secret: str, timeout=10):
    data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    r = requests.post(TOKEN_URL, data=data, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Token request failed {r.status_code}: {r.text[:300]}")
    j = r.json()
    return j.get("access_token"), int(j.get("expires_in", 0))


def fetch_states_timepoints_oauth(client_id, client_secret, times, bbox,
                                 sleep_per_call=0.05, timeout=15, max_retries=2):
    lat_min, lat_max, lon_min, lon_max = bbox
    records = []

    token, ttl = obtain_token(client_id, client_secret)
    token_acquired_at = time.time()
    headers = {"Authorization": f"Bearer {token}"}

    for t in tqdm(times, desc="fetching times"):
        # token refresh
        if time.time() - token_acquired_at > max(0, ttl - 30):
            token, ttl = obtain_token(client_id, client_secret)
            token_acquired_at = time.time()
            headers = {"Authorization": f"Bearer {token}"}

        params = {"lamin": lat_min, "lamax": lat_max, "lomin": lon_min, "lomax": lon_max, "time": int(t)}

        attempt = 0
        while attempt <= max_retries:
            attempt += 1
            try:
                resp = requests.get(STATES_URL, params=params, headers=headers, timeout=timeout)
            except Exception as e:
                print(f"[WARN] request exception t={t}: {e}")
                time.sleep(1.0)
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception as e:
                    print(f"[WARN] invalid json t={t}: {e}")
                    data = None

                if data and data.get("states"):
                    for s in data["states"]:
                        lon = s[5]
                        lat = s[6]
                        if lat is None or lon is None:
                            continue
                        records.append({
                            "time": int(t),
                            "icao24": s[0],
                            "callsign": (s[1].strip() if s[1] else ""),
                            "lat": lat,
                            "lon": lon,
                            "baro_altitude": s[7],
                            "velocity": s[9],
                            "heading": s[10],
                            "vertical_rate": s[11] if len(s) > 11 else None,
                        })
                break

            if resp.status_code == 401:
                token, ttl = obtain_token(client_id, client_secret)
                token_acquired_at = time.time()
                headers = {"Authorization": f"Bearer {token}"}
                continue

            if resp.status_code in (429, 503):
                backoff = 1.0 + attempt * 1.5
                print(f"[WARN] HTTP {resp.status_code} t={t} backoff={backoff}s")
                time.sleep(backoff)
                continue

            print(f"[WARN] HTTP {resp.status_code} t={t}: {resp.text[:200]}")
            break

        time.sleep(sleep_per_call)

    if records:
        return pd.DataFrame.from_records(records)
    return pd.DataFrame(columns=["time", "icao24", "callsign", "lat", "lon",
                                 "baro_altitude", "velocity", "heading", "vertical_rate"])


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    need_cols = ["time", "icao24", "callsign", "lat", "lon",
                 "baro_altitude", "velocity", "heading", "vertical_rate"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = None
    df = df[need_cols].copy()

    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
    df["icao24"] = df["icao24"].astype(str)
    df["callsign"] = df["callsign"].fillna("").astype(str)
    for c in ["lat", "lon", "baro_altitude", "velocity", "heading", "vertical_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time", "icao24", "lat", "lon"])
    df["time"] = df["time"].astype(int)

    # 去重：同一 (time,icao24) 保留最后一次
    df = df.sort_values(["time", "icao24"]).drop_duplicates(["time", "icao24"], keep="last")
    return df


def append_dedup_csv_gz(path: str, df_new: pd.DataFrame):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_old = normalize_df(df_old)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = normalize_df(df_all)
    else:
        df_all = df_new

    df_all.to_csv(path, index=False, compression="gzip")
    return len(df_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_id", default=None)
    ap.add_argument("--client_secret", default=None)
    ap.add_argument("--secrets_json", default=None, help="可选：从本地 JSON 读取 client_id/client_secret")

    ap.add_argument("--out_root", required=True)

    ap.add_argument("--duration_minutes", type=int, default=40)
    ap.add_argument("--time_step", type=int, default=10)
    ap.add_argument("--end_offset_s", type=int, default=30)

    ap.add_argument("--lat_min", type=float, default=22.5)
    ap.add_argument("--lat_max", type=float, default=24.0)
    ap.add_argument("--lon_min", type=float, default=112.5)
    ap.add_argument("--lon_max", type=float, default=114.5)

    ap.add_argument("--max_altitude_m", type=float, default=3000.0)
    ap.add_argument("--sleep_per_call", type=float, default=0.05)
    args = ap.parse_args()

    client_id = args.client_id
    client_secret = args.client_secret
    if (not client_id or not client_secret) and args.secrets_json:
        client_id, client_secret = load_secrets(args.secrets_json)

    if not client_id or not client_secret:
        raise RuntimeError("请通过 --client_id/--client_secret 传入密钥，或使用 --secrets_json secrets.json")

    # 用 UTC 日期分桶
    utc_now = datetime.now(timezone.utc)
    day_dir = os.path.join(args.out_root, utc_now.strftime("%Y%m%d"))
    os.makedirs(day_dir, exist_ok=True)

    end_time = int(time.time()) - args.end_offset_s
    start_time = end_time - int(args.duration_minutes) * 60
    times = list(range(start_time, end_time + 1, args.time_step))

    print(f"[INFO] saving to: {day_dir}")
    print(f"[INFO] fetching {len(times)} snapshots from {datetime.utcfromtimestamp(start_time)} to {datetime.utcfromtimestamp(end_time)} (UTC)")

    df_raw = fetch_states_timepoints_oauth(
        client_id, client_secret, times,
        (args.lat_min, args.lat_max, args.lon_min, args.lon_max),
        sleep_per_call=args.sleep_per_call
    )
    if df_raw is None or len(df_raw) == 0:
        print("[WARN] no states returned; exit.")
        return

    df_raw = normalize_df(df_raw)
    df_low = df_raw[df_raw["baro_altitude"].notnull() & (df_raw["baro_altitude"] < args.max_altitude_m)].copy()
    df_low = normalize_df(df_low)

    raw_path = os.path.join(day_dir, "opensky_raw_states.csv.gz")
    low_path = os.path.join(day_dir, "opensky_lowalt_states.csv.gz")

    raw_total = append_dedup_csv_gz(raw_path, df_raw)
    low_total = append_dedup_csv_gz(low_path, df_low)

    summary = {
        "utc_day": utc_now.strftime("%Y%m%d"),
        "utc_start": int(start_time),
        "utc_end": int(end_time),
        "bbox": [args.lat_min, args.lat_max, args.lon_min, args.lon_max],
        "time_step": args.time_step,
        "duration_minutes": args.duration_minutes,
        "raw_new_rows": int(len(df_raw)),
        "low_new_rows": int(len(df_low)),
        "raw_total_rows": int(raw_total),
        "low_total_rows": int(low_total),
        "max_altitude_m": float(args.max_altitude_m),
    }
    with open(os.path.join(day_dir, "fetch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] raw_total={raw_total} low_total={low_total}")


if __name__ == "__main__":
    main()
