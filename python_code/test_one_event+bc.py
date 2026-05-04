#!/usr/bin/env python3
"""
Test ONE event to compare with Fortran results.

You can either:
1. Enter event parameters manually
2. Use a single-line CSV file

Usage:
    python test_one_event.py
    
Then enter your event details when prompted.
"""

import os
import numpy as np
from datetime import datetime

# Check obspy
try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
    OBSPY_OK = True
except ImportError:
    OBSPY_OK = False
    print("ERROR: obspy not installed!")
    exit(1)

from depth_bins import get_depth_bin, print_depth_bin_info
from travel_time import JBTables


def test_single_event():
    """
    Test a single event interactively.
    """
    print("\n" + "="*70)
    print("TEST SINGLE EVENT - Compare with Fortran")
    print("="*70)
    
    print("\nEnter event parameters (or press Enter for defaults):\n")
    
    # Get event parameters
    event_id = input("Event ID [TEST_EVENT]: ").strip() or "TEST_EVENT"
    
    date_str = input("Date (YYYY-MM-DD) [2023-01-15]: ").strip() or "2023-01-15"
    time_str = input("Time (HH:MM:SS) [12:30:00]: ").strip() or "12:30:00"
    
    lat = float(input("Latitude [-15.0]: ").strip() or "-15.0")
    lon = float(input("Longitude [-75.0]: ").strip() or "-75.0")
    depth = float(input("Depth (km) [33.0]: ").strip() or "33.0")
    
    mag = float(input("Magnitude [7.0]: ").strip() or "7.0")
    moment_input = input("Moment (dyn-cm) [auto from mag]: ").strip()
    if moment_input:
        moment = float(moment_input)
    else:
        moment = 10 ** (1.5 * mag + 16.1)
    
    strike = float(input("Strike (degrees) [0.0]: ").strip() or "0.0")
    dip = float(input("Dip (degrees) [45.0]: ").strip() or "45.0")
    rake = float(input("Rake (degrees) [90.0]: ").strip() or "90.0")
    
    # Parse datetime
    origin_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
    
    print("\n" + "-"*70)
    print("EVENT PARAMETERS:")
    print("-"*70)
    print(f"  Event ID: {event_id}")
    print(f"  Origin:   {origin_time}")
    print(f"  Location: ({lat}, {lon})")
    print(f"  Depth:    {depth} km")
    print(f"  Magnitude: {mag}")
    print(f"  Moment:   {moment:.3e} dyn-cm")
    print(f"  Mechanism: strike={strike}, dip={dip}, rake={rake}")
    
    # Show depth bin
    print_depth_bin_info(depth)
    
    # Ask for processing options
    print("\nProcessing options:")
    networks = input("Networks [II,IU]: ").strip() or "II,IU"
    channel = input("Channel [BHZ]: ").strip() or "BHZ"
    min_dist = float(input("Min distance (deg) [30.0]: ").strip() or "30.0")
    max_dist = float(input("Max distance (deg) [80.0]: ").strip() or "80.0")
    
    # Create output directory
    output_dir = f"./test_{event_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("DOWNLOADING DATA FROM IRIS...")
    print("="*70)
    
    # Initialize client
    client = Client("IRIS")
    
    # Time window
    starttime = UTCDateTime(origin_time) - 60  # 1 min before
    endtime = UTCDateTime(origin_time) + 1800   # 30 min after
    
    # Get stations
    print(f"\nSearching for stations ({networks}, {channel}, {min_dist}-{max_dist}°)...")
    
    try:
        inventory = client.get_stations(
            network=networks,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
            latitude=lat,
            longitude=lon,
            minradius=min_dist,
            maxradius=max_dist,
            level="response"
        )
    except Exception as e:
        print(f"ERROR getting stations: {e}")
        return
    
    n_stations = sum(len(net) for net in inventory)
    print(f"Found {n_stations} stations")
    
    if n_stations == 0:
        print("No stations found! Try adjusting distance range.")
        return
    
    # Download and process each station
    results = []
    jb_tables = JBTables()
    
    print("\n" + "-"*70)
    print("PROCESSING STATIONS:")
    print("-"*70)
    print(f"{'Station':<12} {'Dist':>8} {'Az':>8} {'Slowness':>10} {'Θ':>8}")
    print("-"*70)
    
    for network in inventory:
        for station in network:
            sta_code = f"{network.code}.{station.code}"
            
            try:
                # Download waveform
                st = client.get_waveforms(
                    network=network.code,
                    station=station.code,
                    location="*",
                    channel=channel,
                    starttime=starttime,
                    endtime=endtime
                )
                
                if len(st) == 0:
                    continue
                
                # Get coordinates
                sta_lat = station.latitude
                sta_lon = station.longitude
                
                # Calculate distance and azimuth
                dist_m, az, baz = gps2dist_azimuth(lat, lon, sta_lat, sta_lon)
                dist_deg = kilometers2degrees(dist_m / 1000.0)
                
                if dist_deg < min_dist or dist_deg > max_dist:
                    continue
                
                # Get slowness
                slowness = jb_tables.get_slowness(dist_deg, depth)
                
                # Remove instrument response
                st.remove_response(inventory=inventory, output="VEL")
                
                # Get data
                tr = st[0]
                data = tr.data.astype(float)
                dt = tr.stats.delta
                
                # Compute energy (simplified for testing)
                from instrument_response import InstrumentResponse
                from energy_calculation import compute_seismic_energy
                
                # Create flat response (already removed)
                instrument = InstrumentResponse(
                    a0=1.0,
                    sensitivity=1.0,
                    zeros=np.array([]),
                    poles=np.array([]),
                    unit='M/S'
                )
                
                result = compute_seismic_energy(
                    data=data,
                    dt=dt,
                    instrument=instrument,
                    elat=lat, elon=lon,
                    slat=sta_lat, slon=sta_lon,
                    depth_km=depth,
                    moment=moment,
                    strike=strike, dip=dip, rake=rake,
                    tmin=0.5, tmax=10.0,
                    station_code=sta_code,
                    jb_tables=jb_tables
                )
                
                results.append(result)
                print(f"{sta_code:<12} {dist_deg:>8.2f} {az:>8.1f} {slowness:>10.4f} {result.theta_estimated:>8.2f}")
                
            except Exception as e:
                print(f"{sta_code:<12} ERROR: {str(e)[:40]}")
                continue
    
    # Summary
    if results:
        theta_values = [r.theta_estimated for r in results]
        theta_mean = np.mean(theta_values)
        theta_std = np.std(theta_values)
        
        print("-"*70)
        print(f"{'AVERAGE':<12} {'':>8} {'':>8} {'':>10} {theta_mean:>8.2f} ± {theta_std:.2f}")
        print("="*70)
        
        print("\n" + "="*70)
        print("FINAL RESULT:")
        print("="*70)
        print(f"  Stations processed: {len(results)}")
        print(f"  Average Θ: {theta_mean:.2f} ± {theta_std:.2f}")
        print(f"  Depth bin: {results[0].depth_bin}")
        print("="*70)
        
        # Save results
        result_file = os.path.join(output_dir, "results.txt")
        with open(result_file, 'w') as f:
            f.write(f"Event: {event_id}\n")
            f.write(f"Origin: {origin_time}\n")
            f.write(f"Location: ({lat}, {lon}), depth={depth} km\n")
            f.write(f"Moment: {moment:.3e} dyn-cm\n")
            f.write(f"Mechanism: strike={strike}, dip={dip}, rake={rake}\n")
            f.write(f"\nStations: {len(results)}\n")
            f.write(f"Average Theta: {theta_mean:.2f} ± {theta_std:.2f}\n\n")
            f.write(f"{'Station':<12} {'Dist':>8} {'Az':>8} {'Theta':>8}\n")
            f.write("-"*40 + "\n")
            for r in results:
                f.write(f"{r.station:<12} {r.distance_deg:>8.2f} {r.azimuth:>8.1f} {r.theta_estimated:>8.2f}\n")
        
        print(f"\nResults saved to: {result_file}")
        
        print("\n" + "="*70)
        print("COMPARE WITH YOUR FORTRAN RESULT:")
        print("="*70)
        print(f"  Python Θ = {theta_mean:.2f}")
        print("  Fortran Θ = ??? (enter your value)")
        print("="*70)
    else:
        print("\nNo stations processed successfully!")


if __name__ == "__main__":
    test_single_event()



