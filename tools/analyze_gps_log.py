#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze GPS or GPS+IMU CSV logs.

For a moving vehicle, the script reports path length, displacement and bounding
box. For a static test, the radius metrics around the mean position estimate
the observed drift/error spread.
"""

import argparse
import csv
import math
from statistics import mean
from typing import Iterable, List, Optional, Tuple


EARTH_RADIUS_M = 6378137.0


def enu_from_latlon(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    phi = math.radians(lat)
    phi0 = math.radians(lat0)
    lam = math.radians(lon)
    lam0 = math.radians(lon0)
    east = EARTH_RADIUS_M * (lam - lam0) * math.cos((phi + phi0) / 2.0)
    north = EARTH_RADIUS_M * (phi - phi0)
    return east, north


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pick_columns(fieldnames: Iterable[str], source: str) -> Tuple[str, str]:
    fields = set(fieldnames)
    if source == "fused":
        return "fused_lat", "fused_lon"
    if source == "gps":
        if "gps_lat" in fields and "gps_lon" in fields:
            return "gps_lat", "gps_lon"
        return "lat", "lon"
    if "fused_lat" in fields and "fused_lon" in fields:
        return "fused_lat", "fused_lon"
    if "lat" in fields and "lon" in fields:
        return "lat", "lon"
    if "gps_lat" in fields and "gps_lon" in fields:
        return "gps_lat", "gps_lon"
    raise ValueError("log does not contain recognizable latitude/longitude columns")


def load_points(path: str, source: str) -> Tuple[List[dict], str, str]:
    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError("empty CSV log")
        lat_col, lon_col = pick_columns(reader.fieldnames, source)
        rows = []
        for row in reader:
            lat = parse_float(row.get(lat_col))
            lon = parse_float(row.get(lon_col))
            if lat is None or lon is None:
                continue
            row["_lat"] = lat
            row["_lon"] = lon
            rows.append(row)
    return rows, lat_col, lon_col


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GPS/GPS+IMU CSV trajectory logs")
    parser.add_argument("csv_log", help="CSV log path")
    parser.add_argument("--source", choices=("auto", "fused", "gps"), default="auto",
                        help="which coordinate columns to analyze")
    parser.add_argument("--reference", nargs=2, type=float, metavar=("LAT", "LON"),
                        help="known reference coordinate for error statistics")
    args = parser.parse_args()

    rows, lat_col, lon_col = load_points(args.csv_log, args.source)
    if len(rows) < 2:
        raise SystemExit("not enough valid coordinate rows")

    lats = [row["_lat"] for row in rows]
    lons = [row["_lon"] for row in rows]
    ref = tuple(args.reference) if args.reference else (mean(lats), mean(lons))
    enu = [enu_from_latlon(row["_lat"], row["_lon"], ref[0], ref[1]) for row in rows]

    step_lengths = [
        math.hypot(enu[i][0] - enu[i - 1][0], enu[i][1] - enu[i - 1][1])
        for i in range(1, len(enu))
    ]
    radii = [math.hypot(e, n) for e, n in enu]
    displacement = math.hypot(enu[-1][0] - enu[0][0], enu[-1][1] - enu[0][1])
    rms = math.sqrt(mean([r * r for r in radii]))

    elapsed_values = [parse_float(row.get("elapsed_s")) for row in rows]
    elapsed_values = [v for v in elapsed_values if v is not None]
    duration = max(elapsed_values) - min(elapsed_values) if len(elapsed_values) >= 2 else None

    print(f"file={args.csv_log}")
    print(f"source_columns={lat_col},{lon_col} points={len(rows)}")
    if duration is not None:
        print(f"duration={duration:.2f}s")
    print(f"first_lat={lats[0]:.8f} first_lon={lons[0]:.8f}")
    print(f"last_lat={lats[-1]:.8f} last_lon={lons[-1]:.8f}")
    print(f"reference_lat={ref[0]:.8f} reference_lon={ref[1]:.8f}")
    print(f"path_length={sum(step_lengths):.2f}m displacement={displacement:.2f}m")
    print(
        "bbox="
        f"E[{min(e for e, _ in enu):+.2f},{max(e for e, _ in enu):+.2f}]m "
        f"N[{min(n for _, n in enu):+.2f},{max(n for _, n in enu):+.2f}]m"
    )
    label = "error" if args.reference else "drift_from_mean"
    print(
        f"{label}: rms={rms:.2f}m p50={percentile(radii, 50):.2f}m "
        f"p95={percentile(radii, 95):.2f}m max={max(radii):.2f}m"
    )


if __name__ == "__main__":
    main()
