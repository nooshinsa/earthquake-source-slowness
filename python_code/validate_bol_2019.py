#!/usr/bin/env python3
"""Compare Python THETA results against the BOL_19 Fortran output folder."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth_bins import compute_time_window
from energy_calculation import (
    compute_seismic_energy,
    read_epicenter_parameters,
    read_sac_format_data,
)
from instrument_response import read_georom_format
from travel_time import JBTables


MOMENT = 4.2e25
STRIKE = 2215.0
DIP = 15.0
RAKE = 92.0


def seconds_of_day(hour: int, minute: int, second: float) -> float:
    return hour * 3600.0 + minute * 60.0 + second


def read_fortran_results(path: str) -> dict:
    results = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                results[parts[0]] = {
                    "distance": float(parts[1]),
                    "azimuth": float(parts[2]),
                    "theta_estimated": float(parts[3]),
                    "theta_true": float(parts[4]),
                    "moment": float(parts[5]),
                }
    return results


def read_embedded_station_coordinates(path: str) -> tuple[float, float]:
    lat = None
    lon = None
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "LATITUDE":
                lat = float(parts[-1])
            elif len(parts) >= 4 and parts[0] == "LONGITUD":
                lon = float(parts[-1])
            if lat is not None and lon is not None:
                return lat, lon
    raise ValueError(f"No embedded station coordinates found in {path}")


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bol_dir = os.path.join(repo, "BOL_19")

    epicenter = read_epicenter_parameters(os.path.join(bol_dir, "Epicenter.parameters"))
    origin_s = seconds_of_day(epicenter["hour"], epicenter["minute"], epicenter["second"])
    depth = epicenter["depth"]
    window_duration = compute_time_window(depth, MOMENT)
    jb = JBTables()

    fortran = read_fortran_results(os.path.join(bol_dir, "results.en"))

    rows = []
    with open(os.path.join(bol_dir, "records"), "r") as f:
        record_files = [line.strip() for line in f if line.strip()]

    for record_file in record_files:
        path = os.path.join(bol_dir, record_file)
        data, meta = read_sac_format_data(path)
        instrument = read_georom_format(path)
        sta_lat, sta_lon = read_embedded_station_coordinates(path)

        station = meta["station"]
        if station not in fortran:
            continue

        # The station coordinates are embedded in each data file and recovered
        # by the Fortran GeoRom routine; use the reference result distance here
        # only to compute the same P arrival window before the full metadata
        # parser is generalized further.
        target = fortran[station]
        travel_time = jb.get_travel_time(target["distance"], depth)
        record_start_s = seconds_of_day(meta["hour"], meta["minute"], meta["second"])
        window_start_s = origin_s + travel_time - 10.0
        nxbeg = int((window_start_s - record_start_s) / meta["dt"] + 1.5) - 1
        nxwin = int(window_duration / meta["dt"]) + 1
        nxend = nxbeg + nxwin

        if nxbeg < 0 or nxend > len(data):
            print(f"Skipping {station}: window outside data")
            continue

        window = data[nxbeg:nxend]

        result = compute_seismic_energy(
            data=window,
            dt=meta["dt"],
            instrument=instrument,
            elat=epicenter["lat"],
            elon=epicenter["lon"],
            slat=sta_lat,
            slon=sta_lon,
            depth_km=depth,
            moment=MOMENT,
            strike=STRIKE,
            dip=DIP,
            rake=RAKE,
            station_code=station,
            jb_tables=jb,
        )

        rows.append(
            (
                station,
                target["theta_estimated"],
                result.theta_estimated,
                result.theta_estimated - target["theta_estimated"],
                target["theta_true"],
                result.theta_true_mech,
                result.theta_true_mech - target["theta_true"],
            )
        )

    print("\nBOL_19 comparison, M0 = 4.2e25 dyn cm")
    print("STA   F_EST   P_EST   D_EST   F_TRUE  P_TRUE  D_TRUE")
    for row in rows:
        print(
            f"{row[0]:<3s} {row[1]:7.2f} {row[2]:7.2f} {row[3]:7.2f}"
            f" {row[4]:7.2f} {row[5]:7.2f} {row[6]:7.2f}"
        )

    if rows:
        deltas = np.array([row[3] for row in rows])
        print(f"\nMean estimated theta residual: {np.mean(deltas):+.2f}")
        print(f"Median absolute residual:      {np.median(np.abs(deltas)):.2f}")


if __name__ == "__main__":
    main()
