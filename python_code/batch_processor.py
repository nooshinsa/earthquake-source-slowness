#!/usr/bin/env python3
"""
Batch Processor for IRIS Data

Processes multiple stations from IRIS (II and IU networks) and computes
average Theta (Θ) across all stations.

Typical workflow:
1. Download BHZ data from IRIS for II/IU networks at 30-80° distance
2. Run this processor on all station files
3. Get average Θ value for the event

Based on the Fortran newenergytable.f and processenerg.f workflow.
"""

import os
import glob
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from energy_calculation import compute_seismic_energy, EnergyResult, classify_event
from instrument_response import InstrumentResponse, read_georom_format
from travel_time import JBTables
from depth_bins import print_depth_bin_info, get_depth_bin


@dataclass
class EventSummary:
    """
    Summary of Theta calculation for an event across multiple stations.
    
    Attributes
    ----------
    event_id : str
        Event identifier
    n_stations : int
        Number of stations processed successfully
    theta_estimated_mean : float
        Mean Θ (estimated focal mechanism)
    theta_estimated_std : float
        Standard deviation of Θ (estimated)
    theta_true_mean : float
        Mean Θ (true focal mechanism)
    theta_true_std : float
        Standard deviation of Θ (true)
    station_results : list
        Individual station results
    """
    event_id: str
    n_stations: int
    theta_estimated_mean: float
    theta_estimated_std: float
    theta_true_mean: float
    theta_true_std: float
    station_results: List[EnergyResult]


def process_iris_event(
    data_dir: str,
    epicenter_lat: float,
    epicenter_lon: float,
    depth_km: float,
    moment: float,
    strike: float = 0.0,
    dip: float = 45.0,
    rake: float = 90.0,
    min_distance: float = 30.0,
    max_distance: float = 80.0,
    event_id: str = "EVENT",
    output_file: str = "results.en"
) -> EventSummary:
    """
    Process all IRIS station files for an event and compute average Theta.
    
    This replicates the Fortran workflow of processing multiple stations
    and averaging the results.
    
    Parameters
    ----------
    data_dir : str
        Directory containing station data files
    epicenter_lat, epicenter_lon : float
        Epicenter coordinates (degrees)
    depth_km : float
        Source depth in km
    moment : float
        Seismic moment in dyn-cm
    strike, dip, rake : float
        Focal mechanism parameters (degrees)
    min_distance, max_distance : float
        Distance range to include (degrees)
    event_id : str
        Event identifier for output
    output_file : str
        Output file for results
        
    Returns
    -------
    EventSummary
        Summary with average Theta and individual station results
    """
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {event_id}")
    print(f"{'='*70}")
    print(f"Epicenter: ({epicenter_lat}°, {epicenter_lon}°)")
    print(f"Depth: {depth_km} km")
    
    # Show depth bin info
    depth_bin = get_depth_bin(depth_km)
    print(f"Depth bin: {depth_bin.name}")
    print(f"Moment: {moment:.2e} dyn-cm")
    print(f"Distance range: {min_distance}° - {max_distance}°")
    print(f"{'='*70}\n")
    
    # Initialize JB tables once
    jb_tables = JBTables()
    
    # Find all data files
    data_files = find_data_files(data_dir)
    print(f"Found {len(data_files)} data files to process\n")
    
    results = []
    skipped = []
    
    for i, (data_file, resp_file, station_info) in enumerate(data_files):
        print(f"\n[{i+1}/{len(data_files)}] Processing {station_info['station']}...")
        
        try:
            # Read data
            data, dt, slat, slon = read_iris_data(data_file)
            
            # Read instrument response
            instrument = read_instrument_response(resp_file)
            
            # Compute distance first to check if in range
            from seismic_utils import great_circle
            dist_deg, _, _, _ = great_circle(epicenter_lat, epicenter_lon, slat, slon)
            
            if dist_deg < min_distance or dist_deg > max_distance:
                print(f"   Skipping: distance {dist_deg:.1f}° outside range")
                skipped.append((station_info['station'], f"distance {dist_deg:.1f}°"))
                continue
            
            # Compute energy
            result = compute_seismic_energy(
                data=data,
                dt=dt,
                instrument=instrument,
                elat=epicenter_lat, elon=epicenter_lon,
                slat=slat, slon=slon,
                depth_km=depth_km,
                moment=moment,
                strike=strike, dip=dip, rake=rake,
                tmin=0.5, tmax=10.0,
                station_code=station_info['station'],
                jb_tables=jb_tables
            )
            
            results.append(result)
            print(f"   ✓ Θ = {result.theta_estimated:.2f}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            skipped.append((station_info.get('station', 'unknown'), str(e)))
            continue
    
    # Compute statistics
    if len(results) == 0:
        print("\n⚠ No stations processed successfully!")
        return EventSummary(
            event_id=event_id,
            n_stations=0,
            theta_estimated_mean=np.nan,
            theta_estimated_std=np.nan,
            theta_true_mean=np.nan,
            theta_true_std=np.nan,
            station_results=[]
        )
    
    theta_est = np.array([r.theta_estimated for r in results])
    theta_true = np.array([r.theta_true_mech for r in results])
    
    summary = EventSummary(
        event_id=event_id,
        n_stations=len(results),
        theta_estimated_mean=np.mean(theta_est),
        theta_estimated_std=np.std(theta_est),
        theta_true_mean=np.mean(theta_true),
        theta_true_std=np.std(theta_true),
        station_results=results
    )
    
    # Write results file (like Fortran results.en)
    write_results_file(output_file, summary)
    
    # Print summary
    print_summary(summary, skipped)
    
    return summary


def find_data_files(data_dir: str) -> List[Tuple[str, str, dict]]:
    """
    Find data files and their corresponding response files.
    
    Looks for common IRIS data formats.
    
    Returns list of (data_file, response_file, station_info) tuples.
    """
    files = []
    
    # Look for SAC files
    sac_files = glob.glob(os.path.join(data_dir, "*.SAC")) + \
                glob.glob(os.path.join(data_dir, "*.sac"))
    
    for sac_file in sac_files:
        base = os.path.splitext(sac_file)[0]
        
        # Try to find response file
        resp_candidates = [
            base + ".resp",
            base + ".RESP",
            base + "_resp.txt",
            os.path.join(data_dir, "RESP." + os.path.basename(base)),
        ]
        
        resp_file = None
        for resp in resp_candidates:
            if os.path.exists(resp):
                resp_file = resp
                break
        
        # Extract station info from filename
        station_info = parse_filename(os.path.basename(sac_file))
        
        files.append((sac_file, resp_file, station_info))
    
    # Also look for our custom format files
    custom_files = glob.glob(os.path.join(data_dir, "*.dat"))
    for dat_file in custom_files:
        base = os.path.splitext(dat_file)[0]
        resp_file = base + "_resp.dat"
        if not os.path.exists(resp_file):
            resp_file = None
        station_info = parse_filename(os.path.basename(dat_file))
        files.append((dat_file, resp_file, station_info))
    
    return files


def parse_filename(filename: str) -> dict:
    """
    Parse station info from filename.
    
    Common formats:
    - IU.ANMO.00.BHZ.SAC
    - II.AAK.10.BHZ.2023.001.sac
    """
    info = {'station': 'UNK', 'network': '', 'channel': '', 'location': ''}
    
    parts = filename.replace('.SAC', '').replace('.sac', '').replace('.dat', '').split('.')
    
    if len(parts) >= 2:
        info['network'] = parts[0]
        info['station'] = parts[1]
    if len(parts) >= 3:
        info['location'] = parts[2]
    if len(parts) >= 4:
        info['channel'] = parts[3]
    
    return info


def read_iris_data(filename: str) -> Tuple[np.ndarray, float, float, float]:
    """
    Read seismic data from IRIS format file.
    
    Supports:
    - SAC binary format (requires obspy)
    - SAC ASCII format
    - Simple text format
    
    Returns: (data, dt, station_lat, station_lon)
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in ['.sac']:
        try:
            # Try using obspy for SAC files
            from obspy import read
            st = read(filename)
            tr = st[0]
            data = tr.data
            dt = tr.stats.delta
            slat = tr.stats.sac.get('stla', 0.0)
            slon = tr.stats.sac.get('stlo', 0.0)
            return data, dt, slat, slon
        except ImportError:
            # Fall back to simple SAC reader
            return read_sac_simple(filename)
    else:
        # Simple text format
        return read_text_data(filename)


def read_sac_simple(filename: str) -> Tuple[np.ndarray, float, float, float]:
    """
    Simple SAC binary reader (without obspy).
    """
    import struct
    
    with open(filename, 'rb') as f:
        # SAC header is 632 bytes
        header = f.read(632)
        
        # Extract key values from header
        # delta (dt) is at offset 0 (float)
        dt = struct.unpack('f', header[0:4])[0]
        
        # stla (station lat) at offset 31*4 = 124
        stla = struct.unpack('f', header[124:128])[0]
        
        # stlo (station lon) at offset 32*4 = 128
        stlo = struct.unpack('f', header[128:132])[0]
        
        # npts (number of points) at offset 79*4 = 316 (integer)
        npts = struct.unpack('i', header[316:320])[0]
        
        # Read data
        data = np.frombuffer(f.read(npts * 4), dtype=np.float32)
    
    # Handle undefined values (-12345.0 in SAC)
    if stla < -1000:
        stla = 0.0
    if stlo < -1000:
        stlo = 0.0
    
    return data, dt, stla, stlo


def read_text_data(filename: str) -> Tuple[np.ndarray, float, float, float]:
    """
    Read simple text format data.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # First line: header info
    # Second line: nx, dt
    # Rest: data
    
    header = lines[0].strip().split()
    nx_dt = lines[1].strip().split()
    nx = int(float(nx_dt[0]))
    dt = float(nx_dt[1])
    
    # Read data
    data_text = ' '.join(lines[2:])
    data = np.array([float(x) for x in data_text.split()[:nx]])
    
    # Try to get station coordinates from header
    slat, slon = 0.0, 0.0
    
    return data, dt, slat, slon


def read_instrument_response(filename: str) -> InstrumentResponse:
    """
    Read instrument response file.
    """
    if filename is None:
        # Return default broadband response
        return InstrumentResponse(
            a0=5.714e8,
            sensitivity=629.0,
            zeros=np.array([0+0j, 0+0j]),
            poles=np.array([
                -0.01234+0.01234j, -0.01234-0.01234j,
                -39.18+49.12j, -39.18-49.12j
            ]),
            unit='M/S'
        )
    
    try:
        return read_georom_format(filename)
    except:
        # Try RESP format
        from instrument_response import read_resp_file
        return read_resp_file(filename)


def write_results_file(filename: str, summary: EventSummary):
    """
    Write results file in format similar to Fortran output.
    """
    with open(filename, 'w') as f:
        f.write(f"# Event: {summary.event_id}\n")
        f.write(f"# Stations: {summary.n_stations}\n")
        f.write(f"# Average Theta (estimated): {summary.theta_estimated_mean:.2f} +/- {summary.theta_estimated_std:.2f}\n")
        f.write(f"# Average Theta (true mech): {summary.theta_true_mean:.2f} +/- {summary.theta_true_std:.2f}\n")
        f.write(f"#\n")
        f.write(f"# Station   Distance   Azimuth   Theta_est  Theta_true  Depth_bin\n")
        
        for r in summary.station_results:
            f.write(f"{r.station:>8s} {r.distance_deg:10.2f} {r.azimuth:10.2f} "
                    f"{r.theta_estimated:10.2f} {r.theta_true_mech:10.2f} {r.depth_bin:>10s}\n")
    
    print(f"\nResults written to: {filename}")


def print_summary(summary: EventSummary, skipped: list):
    """
    Print summary of batch processing.
    """
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Event: {summary.event_id}")
    print(f"Stations processed: {summary.n_stations}")
    if skipped:
        print(f"Stations skipped: {len(skipped)}")
    
    print(f"\n{'─'*70}")
    print("AVERAGE THETA (Θ):")
    print(f"{'─'*70}")
    print(f"  Estimated (statistical focal mech): Θ = {summary.theta_estimated_mean:.2f} ± {summary.theta_estimated_std:.2f}")
    print(f"  True focal mechanism:               Θ = {summary.theta_true_mean:.2f} ± {summary.theta_true_std:.2f}")
    print(f"{'─'*70}")
    
    classification = classify_event(summary.theta_estimated_mean)
    print(f"\nEvent classification: {classification}")
    
    print(f"\n{'─'*70}")
    print("Individual station results:")
    print(f"{'─'*70}")
    print(f"{'Station':>8s} {'Dist(°)':>8s} {'Az(°)':>8s} {'Θ_est':>8s} {'Θ_true':>8s} {'Bin':>8s}")
    print(f"{'─'*70}")
    
    for r in summary.station_results:
        print(f"{r.station:>8s} {r.distance_deg:8.2f} {r.azimuth:8.2f} "
              f"{r.theta_estimated:8.2f} {r.theta_true_mech:8.2f} {r.depth_bin:>8s}")
    
    print(f"{'─'*70}")
    print(f"{'AVERAGE':>8s} {'':>8s} {'':>8s} "
          f"{summary.theta_estimated_mean:8.2f} {summary.theta_true_mean:8.2f}")
    print("="*70)
    
    if skipped:
        print(f"\nSkipped stations ({len(skipped)}):")
        for sta, reason in skipped[:10]:  # Show first 10
            print(f"  {sta}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")


def demo_batch():
    """
    Demo batch processing with synthetic data.
    """
    print("\n" + "="*70)
    print("DEMO: Batch Processing")
    print("="*70)
    print("\nThis demonstrates processing multiple stations for one event.")
    print("In real usage, you would point to a directory with IRIS data.\n")
    
    # Create synthetic results for demo
    np.random.seed(42)
    
    # Simulate 8 stations
    stations = ['ANMO', 'CCM', 'HRV', 'TUC', 'COR', 'SSPA', 'WVT', 'DWPF']
    distances = [45.2, 52.1, 38.5, 41.8, 55.3, 47.9, 50.2, 43.6]
    azimuths = [315.2, 45.8, 22.1, 298.5, 335.1, 55.2, 78.3, 125.6]
    
    # Simulate theta values with some scatter
    base_theta = -5.2  # Slightly slow event
    theta_est = base_theta + np.random.randn(8) * 0.3
    theta_true = base_theta - 0.1 + np.random.randn(8) * 0.25
    
    results = []
    for i in range(8):
        result = EnergyResult(
            station=stations[i],
            distance_deg=distances[i],
            azimuth=azimuths[i],
            slowness_deg=6.5,
            slowness_km=0.058,
            depth_km=33.0,
            depth_bin='SHALLOW',
            energy_estimated=1.5e22,
            energy_true_mech=1.2e22,
            theta_estimated=theta_est[i],
            theta_true_mech=theta_true[i],
            FgP2=0.95,
            FgP2_estimated=0.92,
            geometric_spreading=0.08,
            receiver_function=1.95
        )
        results.append(result)
    
    summary = EventSummary(
        event_id="2023.045 Peru M7.2",
        n_stations=8,
        theta_estimated_mean=np.mean(theta_est),
        theta_estimated_std=np.std(theta_est),
        theta_true_mean=np.mean(theta_true),
        theta_true_std=np.std(theta_true),
        station_results=results
    )
    
    print_summary(summary, [])
    
    print("\n" + "─"*70)
    print("To process your own IRIS data:")
    print("─"*70)
    print("""
1. Download data from IRIS:
   - Networks: II, IU
   - Channels: BHZ  
   - Distance: 30-80°
   
2. Run batch processor:

   from batch_processor import process_iris_event
   
   summary = process_iris_event(
       data_dir='/path/to/your/data',
       epicenter_lat=-15.0,
       epicenter_lon=-75.0,
       depth_km=33.0,
       moment=1.0e27,
       strike=0.0, dip=45.0, rake=90.0,
       min_distance=30.0,
       max_distance=80.0,
       event_id='Peru_2023'
   )
   
   print(f"Average Theta: {summary.theta_estimated_mean:.2f}")
""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_batch()
    else:
        print("Usage:")
        print("  python batch_processor.py --demo     # Run demo")
        print("")
        print("Or import and use in Python:")
        print("  from batch_processor import process_iris_event")
        print("  summary = process_iris_event(data_dir='...', ...)")

