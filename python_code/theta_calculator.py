#!/usr/bin/env python3
"""
Earthquake Source Slowness (THETA) Calculator

Python implementation of Fortran seismological routines for computing:
- P-wave ray parameter p
- Seismic radiated energy E
- Earthquake source slowness parameter Θ = log10(E/M0)

Based on the methodology of Boatwright and Choy (1986).

Usage:
    python theta_calculator.py --help
    python theta_calculator.py --demo
    python theta_calculator.py --data <file> --response <file> --epicenter <file>

Author: Converted from Fortran code
Date: 2025
"""

import argparse
import csv
import contextlib
import io
import os
import numpy as np
import sys
from datetime import datetime
from typing import Optional

# Import our modules
from seismic_utils import great_circle, find_fft_size, seconds_to_hms
from instrument_response import InstrumentResponse, compute_response_array
from travel_time import JBTables, ray_parameter_to_different_units, compute_radiation_coefficients
from energy_calculation import (
    compute_seismic_energy, EnergyResult, classify_event,
    read_epicenter_parameters, read_sac_format_data
)
from depth_bins import print_depth_bin_info, DEPTH_BIN_SUMMARY
from depth_bins import compute_time_window
from instrument_response import read_georom_format


def print_banner():
    """Print program banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          EARTHQUAKE SOURCE SLOWNESS CALCULATOR                ║
║              THETA Radiated-Energy Analysis                   ║
╠═══════════════════════════════════════════════════════════════╣
║  Computes:                                                    ║
║    • P-wave ray parameter p                                   ║
║    • Radiated seismic energy E                                ║
║    • Source slowness Θ = log10(E/M0)                          ║
╚═══════════════════════════════════════════════════════════════╝
""")


def run_demo():
    """Run demonstration with synthetic data."""
    print_banner()
    print("Running demonstration with synthetic data...")
    print("="*60)
    
    # Create JB tables
    print("\n1. Initializing Jeffreys-Bullen travel time tables...")
    jb = JBTables()
    
    # Test parameters - Peru earthquake to Pasadena
    elat, elon = -15.0, -75.0   # Epicenter (Peru)
    slat, slon = 34.0, -118.0   # Station (Pasadena)
    depth = 33.0                 # km
    moment = 1.0e27              # dyn-cm (Mw ~7.2)
    
    print(f"\n2. Event parameters:")
    print(f"   Epicenter: ({elat}°, {elon}°)")
    print(f"   Depth: {depth} km")
    print(f"   Seismic moment: {moment:.2e} dyn-cm")
    
    # Calculate distance and azimuth
    print("\n3. Computing great circle path...")
    dist_deg, az_es, az_se, gc = great_circle(elat, elon, slat, slon)
    print(f"   Station: ({slat}°, {slon}°)")
    print(f"   Distance: {dist_deg:.2f}°")
    print(f"   Azimuth (event→station): {az_es:.2f}°")
    print(f"   Back azimuth: {az_se:.2f}°")
    
    # Get travel time and ray parameter p.
    print("\n4. Computing P-wave travel time and ray parameter p...")
    travel_time = jb.get_travel_time(dist_deg, depth)
    p_deg = jb.get_ray_parameter(dist_deg, depth)
    
    h, m, s = seconds_to_hms(travel_time)
    print(f"   P-wave travel time: {h:02d}:{m:02d}:{s:05.2f} ({travel_time:.2f} s)")
    
    # Convert ray parameter p to different units
    p_units = ray_parameter_to_different_units(p_deg)
    print(f"\n5. Ray parameter p:")
    print(f"   p = {p_units['p_deg']:.4f} s/deg")
    print(f"   p = {p_units['p_km']:.6f} s/km")
    print(f"   p = {p_units['p_rad']:.2f} s/rad")
    
    # Compute radiation pattern
    print("\n6. Computing radiation pattern (assuming thrust fault)...")
    coeffs = compute_radiation_coefficients(
        distance_deg=dist_deg,
        depth_km=depth,
        azimuth=az_es,
        strike=0.0,    # N-S strike
        dip=45.0,      # 45° dip
        rake=90.0      # Pure thrust
    )
    
    print(f"   Takeoff angle: {coeffs['ih']*180/np.pi:.2f}°")
    print(f"   Geometric spreading (g): {coeffs['g']:.4f}")
    print(f"   Receiver function (rpz): {coeffs['rpz']:.4f}")
    print(f"   Direct P radiation (Fp): {coeffs['Fp']:.4f}")
    print(f"   pP radiation (Fpp): {coeffs['Fpp']:.4f}")
    print(f"   sP radiation (Fsp): {coeffs['Fsp']:.4f}")
    print(f"   PP reflection coeff: {coeffs['PP']:.4f}")
    print(f"   SP reflection coeff: {coeffs['SP']:.4f}")
    
    # Generate synthetic seismogram
    print("\n7. Generating synthetic P-wave seismogram...")
    dt = 0.5  # seconds
    duration = 120  # seconds
    n_samples = int(duration / dt)
    t = np.arange(n_samples) * dt
    
    # Simulate P arrival with realistic wavelet
    p_arrival = 30.0
    data = np.zeros(n_samples)
    for i, ti in enumerate(t):
        if ti > p_arrival and ti < p_arrival + 40:
            tau = ti - p_arrival
            # P-wave coda
            data[i] = 5000 * np.exp(-tau/8) * np.sin(2*np.pi*tau/3)
            # Add some higher frequency content
            data[i] += 2000 * np.exp(-tau/4) * np.sin(2*np.pi*tau/1)
    
    # Add realistic noise
    data += np.random.randn(n_samples) * 50
    
    print(f"   Duration: {duration} s")
    print(f"   Sampling: {dt} s ({1/dt:.1f} Hz)")
    print(f"   Samples: {n_samples}")
    
    # Create instrument response (broadband seismometer)
    print("\n8. Setting up instrument response (broadband seismometer)...")
    instrument = InstrumentResponse(
        a0=5.714e8,
        sensitivity=629.0,
        zeros=np.array([0+0j, 0+0j]),  # Two zeros at origin
        poles=np.array([
            -0.01234+0.01234j,  # Long-period corner
            -0.01234-0.01234j,
            -39.18+49.12j,      # Short-period rolloff
            -39.18-49.12j
        ]),
        unit='M/S'  # Velocity output
    )
    print(f"   Total gain: {instrument.total_gain:.3e}")
    print(f"   Poles: {len(instrument.poles)}, Zeros: {len(instrument.zeros)}")
    print(f"   Output unit: {instrument.unit}")
    
    # Compute energy and theta
    print("\n9. Computing seismic energy and Θ parameter...")
    print("-"*60)
    
    result = compute_seismic_energy(
        data=data,
        dt=dt,
        instrument=instrument,
        elat=elat, elon=elon,
        slat=slat, slon=slon,
        depth_km=depth,
        moment=moment,
        strike=0.0, dip=45.0, rake=90.0,
        tmin=0.5, tmax=10.0,
        station_code="DEMO",
        jb_tables=jb
    )
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"  Station: {result.station}")
    print(f"  Distance: {result.distance_deg:.2f}°")
    print(f"  Azimuth: {result.azimuth:.2f}°")
    print(f"")
    print(f"  Depth: {result.depth_km:.1f} km")
    print(f"  Depth bin: {result.depth_bin}")
    print(f"")
    print(f"  Ray parameter p:")
    print(f"    {result.slowness_deg:.4f} s/deg")
    print(f"    {result.slowness_km:.6f} s/km")
    print(f"")
    print(f"  Geometric factors:")
    print(f"    (FgP)² = {result.FgP2:.4f}")
    print(f"    (FgP)² estimated = {result.FgP2_estimated:.4f} (depth-corrected)")
    print(f"    g = {result.geometric_spreading:.4f}")
    print(f"    rpz = {result.receiver_function:.4f}")
    print(f"")
    print(f"  Energy:")
    print(f"    E (estimated) = {result.energy_estimated:.3e} ergs")
    print(f"    E (true mech) = {result.energy_true_mech:.3e} ergs")
    print(f"")
    print(f"  Theta (Θ = log10(E/M0)):")
    print(f"    Θ estimated = {result.theta_estimated:.2f}")
    print(f"    Θ true mech = {result.theta_true_mech:.2f}")
    print(f"")
    print(f"  Event classification: {classify_event(result.theta_estimated)}")
    print("="*60)
    
    # Interpretation
    print("\nINTERPRETATION:")
    theta = result.theta_estimated
    if theta < -6.0:
        print("  • Very low Θ suggests a SLOW earthquake (tsunami potential)")
        print("  • Long source duration, low rupture velocity")
    elif theta < -5.75:
        print("  • Low Θ indicates a SLOW earthquake")
        print("  • Potential for disproportionate tsunami generation")
    elif theta > -4.3:
        print("  • High Θ indicates a FAST earthquake")
        print("  • High stress drop, short source duration")
    else:
        print("  • Normal Θ value for typical earthquake")
        print("  • Standard rupture characteristics expected")
    
    print("\nReference scale:")
    print("  Θ < -5.75: Tsunami earthquake (slow)")
    print("  -5.75 < Θ < -4.30: Normal earthquake")
    print("  Θ > -4.30: High stress drop (fast)")
    print()


def process_data(data_file: str, response_file: str, epicenter_file: str,
                 moment: float, strike: float, dip: float, rake: float):
    """Process real data file."""
    print_banner()
    print(f"Processing data file: {data_file}")
    print("="*60)
    
    from instrument_response import read_georom_format
    
    # Read epicenter parameters
    print("\nReading epicenter parameters...")
    epi = read_epicenter_parameters(epicenter_file)
    elat = epi['lat']
    elon = epi['lon']
    depth = epi.get('depth', 33.0)
    
    print(f"  Epicenter: ({elat}, {elon})")
    print(f"  Depth: {depth} km")
    
    # Read data
    print("\nReading seismogram data...")
    data, metadata = read_sac_format_data(data_file)
    dt = metadata['dt']
    station = metadata['station']
    slat = metadata.get('lat', 0.0)
    slon = metadata.get('lon', 0.0)
    
    print(f"  Station: {station}")
    print(f"  Samples: {len(data)}")
    print(f"  Sampling: {dt} s")
    
    # Read instrument response
    print("\nReading instrument response...")
    instrument = read_georom_format(response_file)
    print(f"  Gain: {instrument.total_gain:.3e}")
    print(f"  Poles: {instrument.n_poles}, Zeros: {instrument.n_zeros}")
    
    # Compute energy
    print("\nComputing energy and theta...")
    result = compute_seismic_energy(
        data=data,
        dt=dt,
        instrument=instrument,
        elat=elat, elon=elon,
        slat=slat, slon=slon,
        depth_km=depth,
        moment=moment,
        strike=strike, dip=dip, rake=rake,
        tmin=0.5, tmax=10.0,
        station_code=station
    )
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"  Θ = {result.theta_estimated:.2f}")
    print(f"  Classification: {classify_event(result.theta_estimated)}")


def seconds_of_day(hour: int, minute: int, second: float) -> float:
    """Return seconds from midnight."""
    return hour * 3600.0 + minute * 60.0 + second


def read_station_coordinates_from_georom(filename: str) -> tuple[float, float]:
    """Read station latitude/longitude embedded in a Fortran GeoRom data file."""
    lat = None
    lon = None
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "LATITUDE":
                lat = float(parts[-1])
            elif len(parts) >= 4 and parts[0] == "LONGITUD":
                lon = float(parts[-1])
            if lat is not None and lon is not None:
                return lat, lon
    raise ValueError(f"No LATITUDE/LONGITUD response lines found in {filename}")


def read_fortran_theta_results(filename: str) -> dict:
    """Read optional Fortran results.en comparison file."""
    results = {}
    if not filename or not os.path.exists(filename):
        return results

    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                results[parts[0]] = {
                    "fortran_distance_deg": float(parts[1]),
                    "fortran_azimuth": float(parts[2]),
                    "fortran_theta_estimated": float(parts[3]),
                    "fortran_theta_true": float(parts[4]),
                    "fortran_moment": float(parts[5]),
                }
    return results


def run_event_folder(
    folder: str,
    moment: float,
    strike: float,
    dip: float,
    rake: float,
    output: Optional[str] = None,
    compare_file: Optional[str] = None,
    verbose: bool = False,
) -> list:
    """
    Process a Fortran-style THETA event folder.

    The folder should contain:
    - Epicenter.parameters
    - records
    - station files listed in records, each containing waveform and GeoRom response
    """
    folder = os.path.abspath(folder)
    epicenter_path = os.path.join(folder, "Epicenter.parameters")
    records_path = os.path.join(folder, "records")

    if output is None:
        output = os.path.join(folder, "theta_python_results.csv")
    if compare_file is None:
        default_compare = os.path.join(folder, "results.en")
        compare_file = default_compare if os.path.exists(default_compare) else None

    epicenter = read_epicenter_parameters(epicenter_path)
    origin_s = seconds_of_day(
        epicenter.get("hour", 0),
        epicenter.get("minute", 0),
        epicenter.get("second", 0.0),
    )
    depth = epicenter.get("depth", 33.0)
    window_duration = compute_time_window(depth, moment)
    jb_tables = JBTables()
    fortran_results = read_fortran_theta_results(compare_file)

    with open(records_path, "r") as f:
        record_files = [line.strip() for line in f if line.strip()]

    rows = []
    print(f"Processing folder: {folder}")
    print(f"Event: lat={epicenter['lat']}, lon={epicenter['lon']}, depth={depth} km")
    print(f"M0={moment:.3e} dyn cm, mechanism=({strike}, {dip}, {rake})")
    print(f"Records: {len(record_files)}")

    for record_file in record_files:
        path = os.path.join(folder, record_file)
        try:
            data, meta = read_sac_format_data(path)
            instrument = read_georom_format(path)
            sta_lat, sta_lon = read_station_coordinates_from_georom(path)

            distance, azimuth, _, _ = great_circle(
                epicenter["lat"], epicenter["lon"], sta_lat, sta_lon
            )
            travel_time = jb_tables.get_travel_time(distance, depth)
            record_start_s = seconds_of_day(
                meta.get("hour", 0), meta.get("minute", 0), meta.get("second", 0.0)
            )
            window_start_s = origin_s + travel_time - 10.0
            nxbeg = int((window_start_s - record_start_s) / meta["dt"] + 1.5) - 1
            nxwin = int(window_duration / meta["dt"]) + 1
            nxend = nxbeg + nxwin

            if nxbeg < 0 or nxend > len(data):
                raise ValueError(
                    f"P-window outside record: nxbeg={nxbeg}, nxend={nxend}, n={len(data)}"
                )

            window = data[nxbeg:nxend]

            calc = lambda: compute_seismic_energy(
                data=window,
                dt=meta["dt"],
                instrument=instrument,
                elat=epicenter["lat"],
                elon=epicenter["lon"],
                slat=sta_lat,
                slon=sta_lon,
                depth_km=depth,
                moment=moment,
                strike=strike,
                dip=dip,
                rake=rake,
                station_code=meta["station"],
                jb_tables=jb_tables,
            )
            if verbose:
                result = calc()
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = calc()

            row = {
                "station": result.station,
                "record_file": record_file,
                "distance_deg": result.distance_deg,
                "azimuth": result.azimuth,
                "ray_parameter_s_deg": result.slowness_deg,
                "ray_parameter_s_km": result.slowness_km,
                "theta_estimated": result.theta_estimated,
                "theta_true": result.theta_true_mech,
                "energy_estimated_ergs": result.energy_estimated,
                "energy_true_ergs": result.energy_true_mech,
                "depth_bin": result.depth_bin,
                "status": "ok",
                "error": "",
            }

            comparison = fortran_results.get(result.station)
            if comparison:
                row.update(comparison)
                row["theta_estimated_residual"] = (
                    row["theta_estimated"] - comparison["fortran_theta_estimated"]
                )
                row["theta_true_residual"] = (
                    row["theta_true"] - comparison["fortran_theta_true"]
                )

            rows.append(row)
            print(f"  {result.station}: theta={result.theta_estimated:.2f}")

        except Exception as exc:
            station = os.path.basename(record_file)
            rows.append({
                "station": station,
                "record_file": record_file,
                "status": "error",
                "error": str(exc),
            })
            print(f"  {station}: ERROR {exc}")

    fieldnames = [
        "station", "record_file", "distance_deg", "azimuth",
        "ray_parameter_s_deg", "ray_parameter_s_km",
        "theta_estimated", "theta_true",
        "energy_estimated_ergs", "energy_true_ergs",
        "depth_bin", "status", "error",
        "fortran_distance_deg", "fortran_azimuth",
        "fortran_theta_estimated", "fortran_theta_true", "fortran_moment",
        "theta_estimated_residual", "theta_true_residual",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    print(f"\nWrote: {output}")
    print(f"Successful stations: {len(ok_rows)} / {len(rows)}")
    if ok_rows:
        theta = np.array([row["theta_estimated"] for row in ok_rows])
        print(f"Mean estimated theta: {np.mean(theta):.2f} +/- {np.std(theta):.2f}")
        residuals = [
            row["theta_estimated_residual"]
            for row in ok_rows
            if "theta_estimated_residual" in row
        ]
        if residuals:
            residuals = np.array(residuals)
            print(f"Mean Python-Fortran residual: {np.mean(residuals):+.2f}")
            print(f"Median absolute residual:     {np.median(np.abs(residuals)):.2f}")

    return rows


def download_event(args) -> None:
    """Download MiniSEED and StationXML for one manually specified event."""
    try:
        from iris_downloader import CMTEvent, download_event_data
    except ImportError as exc:
        raise SystemExit(
            "download-event requires ObsPy and iris_downloader dependencies. "
            "Try running inside your seismo_env conda environment."
        ) from exc

    event = CMTEvent(
        event_id=args.event_id,
        origin_time=parse_origin_time(args.origin_time),
        latitude=args.latitude,
        longitude=args.longitude,
        depth_km=args.depth,
        magnitude=args.magnitude,
        moment=args.moment,
        strike=args.strike,
        dip=args.dip,
        rake=args.rake,
    )

    files = download_event_data(
        event=event,
        output_dir=args.output_dir,
        networks=args.networks,
        channel=args.channel,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        pre_origin=args.pre_origin,
        post_origin=args.post_origin,
        client_name=args.client,
    )
    print(f"\nDownloaded {len(files)} waveform files.")


def parse_origin_time(value: str) -> datetime:
    """Parse common catalog origin-time strings."""
    text = value.strip()
    normalized = text[:-1] if text.endswith("Z") else text

    for candidate in (normalized, normalized.replace("T", " ")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            pass

    raise SystemExit(
        f"Invalid --origin-time '{value}'. Use a format like "
        "2019-03-15T05:03:50.1 or 2019-03-15T05:03:50."
    )


def read_downloaded_event_info(event_dir: str) -> dict:
    """Read event_info.txt written by download-event."""
    event_file = os.path.join(event_dir, "event_info.txt")
    if not os.path.exists(event_file):
        raise SystemExit(f"Missing event info file: {event_file}")

    info = {}
    with open(event_file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                info[key] = value
    return info


def robust_outlier_mask(values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Return True for values kept by a modified-z-score outlier test."""
    if len(values) == 0:
        return np.array([], dtype=bool)

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.ones(len(values), dtype=bool)

    modified_z = 0.6745 * (values - median) / mad
    return np.abs(modified_z) <= threshold


def process_downloaded_folder(args) -> list:
    """Process a downloaded MiniSEED/StationXML event folder."""
    try:
        from iris_downloader import CMTEvent, process_downloaded_event
    except ImportError as exc:
        raise SystemExit(
            "process-downloaded requires ObsPy. Try running inside "
            "your seismo_env conda environment."
        ) from exc

    event_dir = os.path.abspath(args.folder)
    info = read_downloaded_event_info(event_dir)

    event = CMTEvent(
        event_id=info.get("event_id", os.path.basename(event_dir)),
        origin_time=parse_origin_time(info["origin_time"]),
        latitude=float(info["latitude"]),
        longitude=float(info["longitude"]),
        depth_km=float(info["depth_km"]),
        magnitude=float(info.get("magnitude", 0.0)),
        moment=float(info["moment"]),
        strike=float(info["strike"]),
        dip=float(info["dip"]),
        rake=float(info["rake"]),
    )

    if args.verbose:
        result = process_downloaded_event(event_dir, event)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = process_downloaded_event(event_dir, event)

    rows = []
    for station_result in result["results"]:
        in_distance_range = (
            args.min_distance <= station_result.distance_deg <= args.max_distance
        )
        if not in_distance_range:
            continue

        rows.append({
            "station": station_result.station,
            "distance_deg": station_result.distance_deg,
            "azimuth_deg": station_result.azimuth,
            "ray_parameter_s_deg": station_result.slowness_deg,
            "ray_parameter_s_km": station_result.slowness_km,
            "theta_estimated": station_result.theta_estimated,
            "theta_true_mech": station_result.theta_true_mech,
            "energy_estimated_ergs": station_result.energy_estimated,
            "energy_true_mech_ergs": station_result.energy_true_mech,
            "outlier": False,
            "used_in_mean": True,
        })

    if args.remove_outliers and rows:
        values = np.array([row["theta_estimated"] for row in rows])
        keep_mask = robust_outlier_mask(values, args.outlier_threshold)
        for row, keep in zip(rows, keep_mask):
            row["outlier"] = not bool(keep)
            row["used_in_mean"] = bool(keep)

    if args.output:
        output = args.output
    else:
        output = os.path.join(event_dir, "theta_downloaded_results.csv")

    fieldnames = [
        "station", "distance_deg", "azimuth_deg",
        "ray_parameter_s_deg", "ray_parameter_s_km",
        "theta_estimated", "theta_true_mech",
        "energy_estimated_ergs", "energy_true_mech_ergs",
        "outlier", "used_in_mean",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    used = [row for row in rows if row["used_in_mean"]]
    print(f"Processed downloaded folder: {event_dir}")
    print(f"Distance range: {args.min_distance:.1f}-{args.max_distance:.1f} degrees")
    print(f"Wrote: {output}")
    print(f"Stations in distance range: {len(rows)}")
    if args.remove_outliers:
        print(f"Outliers removed from mean: {len(rows) - len(used)}")
    if used:
        theta = np.array([row["theta_estimated"] for row in used])
        print(f"Mean estimated theta: {np.mean(theta):.2f} +/- {np.std(theta):.2f}")
        print(f"Classification: {classify_event(float(np.mean(theta)))}")
    else:
        print("No stations available for mean theta.")

    return rows


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Earthquake Source Slowness (THETA) Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python theta_calculator.py --demo
  python theta_calculator.py --data record.dat --response resp.dat --epicenter epi.dat
  python theta_calculator.py run-folder ../BOL_19 --moment 4.2e25 --strike 55 --dip 15 --rake 92
  python theta_calculator.py download-event --event-id BOL_2019 --origin-time 2019-03-15T05:03:50.1 --latitude -17.74 --longitude -65.90 --depth 381 --magnitude 6.3 --moment 4.2e25 --strike 55 --dip 15 --rake 92
  python theta_calculator.py process-downloaded downloads_test/BOL_2019 --remove-outliers

For more information, see the documentation.
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with synthetic data')
    parser.add_argument('--data', type=str,
                       help='Path to seismogram data file')
    parser.add_argument('--response', type=str,
                       help='Path to instrument response file')
    parser.add_argument('--epicenter', type=str,
                       help='Path to epicenter parameters file')
    parser.add_argument('--moment', type=float, default=1.0e27,
                       help='Seismic moment in dyn-cm (default: 1e27)')
    parser.add_argument('--strike', type=float, default=0.0,
                       help='Fault strike in degrees (default: 0)')
    parser.add_argument('--dip', type=float, default=45.0,
                       help='Fault dip in degrees (default: 45)')
    parser.add_argument('--rake', type=float, default=90.0,
                       help='Fault rake in degrees (default: 90)')

    subparsers = parser.add_subparsers(dest="command")

    folder_parser = subparsers.add_parser(
        "run-folder",
        help="Process a Fortran-style THETA event folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    folder_parser.add_argument("folder", help="Event folder containing Epicenter.parameters and records")
    folder_parser.add_argument("--moment", type=float, required=True, help="Seismic moment M0 in dyn-cm")
    folder_parser.add_argument("--strike", type=float, required=True, help="Fault strike in degrees")
    folder_parser.add_argument("--dip", type=float, required=True, help="Fault dip in degrees")
    folder_parser.add_argument("--rake", type=float, required=True, help="Fault rake in degrees")
    folder_parser.add_argument("--output", help="Output CSV path")
    folder_parser.add_argument("--compare", help="Optional Fortran results.en file for residuals")
    folder_parser.add_argument("--verbose", action="store_true", help="Print full station diagnostics")

    download_parser = subparsers.add_parser(
        "download-event",
        help="Download MiniSEED waveforms and StationXML responses from an FDSN service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    download_parser.add_argument("--event-id", required=True, help="Event label for output folder")
    download_parser.add_argument("--origin-time", required=True, help="Origin time, e.g. 2019-03-15T05:03:50.1")
    download_parser.add_argument("--latitude", type=float, required=True, help="Event latitude")
    download_parser.add_argument("--longitude", type=float, required=True, help="Event longitude")
    download_parser.add_argument("--depth", type=float, required=True, help="Event depth in km")
    download_parser.add_argument("--magnitude", type=float, default=0.0, help="Event magnitude")
    download_parser.add_argument("--moment", type=float, required=True, help="Seismic moment M0 in dyn-cm")
    download_parser.add_argument("--strike", type=float, required=True, help="Fault strike in degrees")
    download_parser.add_argument("--dip", type=float, required=True, help="Fault dip in degrees")
    download_parser.add_argument("--rake", type=float, required=True, help="Fault rake in degrees")
    download_parser.add_argument("--output-dir", default="theta_results", help="Directory for downloaded data")
    download_parser.add_argument("--networks", default="II,IU", help="Comma-separated network codes")
    download_parser.add_argument("--channel", default="BHZ", help="Channel code")
    download_parser.add_argument("--min-distance", type=float, default=35.0, help="Minimum station distance in degrees")
    download_parser.add_argument("--max-distance", type=float, default=80.0, help="Maximum station distance in degrees")
    download_parser.add_argument("--pre-origin", type=float, default=60.0, help="Seconds before origin to download")
    download_parser.add_argument("--post-origin", type=float, default=1800.0, help="Seconds after origin to download")
    download_parser.add_argument("--client", default="IRIS", help="ObsPy FDSN client name")

    downloaded_parser = subparsers.add_parser(
        "process-downloaded",
        help="Process a downloaded MiniSEED/StationXML event folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    downloaded_parser.add_argument("folder", help="Downloaded event folder containing event_info.txt")
    downloaded_parser.add_argument("--output", help="Output CSV path")
    downloaded_parser.add_argument("--min-distance", type=float, default=35.0, help="Minimum station distance in degrees")
    downloaded_parser.add_argument("--max-distance", type=float, default=80.0, help="Maximum station distance in degrees")
    downloaded_parser.add_argument("--remove-outliers", action="store_true", help="Exclude robust Theta outliers from the reported mean")
    downloaded_parser.add_argument("--outlier-threshold", type=float, default=3.5, help="Modified-z-score cutoff for outlier removal")
    downloaded_parser.add_argument("--verbose", action="store_true", help="Print full station diagnostics")
    
    args = parser.parse_args()
    
    if args.command == "run-folder":
        run_event_folder(
            folder=args.folder,
            moment=args.moment,
            strike=args.strike,
            dip=args.dip,
            rake=args.rake,
            output=args.output,
            compare_file=args.compare,
            verbose=args.verbose,
        )
    elif args.command == "download-event":
        download_event(args)
    elif args.command == "process-downloaded":
        process_downloaded_folder(args)
    elif args.demo:
        run_demo()
    elif args.data and args.response and args.epicenter:
        process_data(
            args.data, args.response, args.epicenter,
            args.moment, args.strike, args.dip, args.rake
        )
    else:
        parser.print_help()
        print("\nUse --demo for a demonstration with synthetic data.")


if __name__ == "__main__":
    main()
