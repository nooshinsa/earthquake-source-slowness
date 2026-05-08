#!/usr/bin/env python3
"""
Test event: 201302060112A SANTA CRUZ ISLANDS
Compare Python vs Fortran Theta values.
"""

import os
import numpy as np
from datetime import datetime

# Event parameters from CMT catalog
EVENT = {
    'id': '201302060112A',
    'name': 'SANTA CRUZ ISLANDS',
    'date': '2013-02-06',
    'time': '01:12:55',
    'lat': -11.18,
    'lon': 165.21,
    'depth_km': 20.2,
    'moment': 9.37e27,  # dyn-cm
    'mw': 7.9,
    'strike': 320,
    'dip': 20,
    'rake': 89
}

print("="*70)
print(f"EVENT: {EVENT['id']} - {EVENT['name']}")
print("="*70)
print(f"Date:     {EVENT['date']} {EVENT['time']} UTC")
print(f"Location: ({EVENT['lat']}, {EVENT['lon']})")
print(f"Depth:    {EVENT['depth_km']} km")
print(f"Mw:       {EVENT['mw']}")
print(f"Moment:   {EVENT['moment']:.2e} dyn-cm")
print(f"Strike/Dip/Rake: {EVENT['strike']}/{EVENT['dip']}/{EVENT['rake']}")
print("="*70)

# Check obspy
try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
except ImportError:
    print("ERROR: obspy not installed!")
    exit(1)

# Import our modules
from depth_bins import get_depth_bin, print_depth_bin_info
from travel_time import JBTables

# Show depth bin info
print("\nDEPTH BIN PARAMETERS:")
print_depth_bin_info(EVENT['depth_km'])

# Initialize
client = Client("IRIS")
jb_tables = JBTables()

# Time window
origin = UTCDateTime(f"{EVENT['date']}T{EVENT['time']}")
starttime = origin - 60  # 1 min before
endtime = origin + 2400  # 40 min after (large event, long window)

# Station parameters
networks = "II,IU"
channel = "BHZ"
min_dist = 30.0
max_dist = 80.0

print(f"\n{'='*70}")
print("SEARCHING FOR STATIONS...")
print(f"{'='*70}")

try:
    inventory = client.get_stations(
        network=networks,
        channel=channel,
        starttime=starttime,
        endtime=endtime,
        latitude=EVENT['lat'],
        longitude=EVENT['lon'],
        minradius=min_dist,
        maxradius=max_dist,
        level="response"
    )
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)

n_stations = sum(len(net) for net in inventory)
print(f"Found {n_stations} stations")

# Process each station
results = []

print(f"\n{'='*70}")
print("PROCESSING STATIONS:")
print(f"{'='*70}")
print(f"{'Station':<15} {'Dist(°)':>8} {'Az(°)':>8} {'p(s/deg)':>10} {'Energy':>12} {'Θ':>8}")
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
                print(f"{sta_code:<15} -- No data")
                continue
            
            # Merge traces if needed
            st.merge(fill_value=0)
            
            # Get station coordinates
            sta_lat = station.latitude
            sta_lon = station.longitude
            
            # Calculate distance and azimuth
            dist_m, az, baz = gps2dist_azimuth(EVENT['lat'], EVENT['lon'], sta_lat, sta_lon)
            dist_deg = kilometers2degrees(dist_m / 1000.0)
            
            if dist_deg < min_dist or dist_deg > max_dist:
                continue
            
            # Get ray parameter p from JB tables
            slowness = jb_tables.get_slowness(dist_deg, EVENT['depth_km'])
            
            # Remove instrument response (convert to velocity m/s)
            st.remove_response(inventory=inventory, output="VEL")
            
            # Get trace data
            tr = st[0]
            data = tr.data.astype(float)
            dt = tr.stats.delta
            fs = tr.stats.sampling_rate
            
            # Import energy calculation
            from energy_calculation import compute_seismic_energy
            from instrument_response import InstrumentResponse
            
            # Response already removed, use flat response
            instrument = InstrumentResponse(
                a0=1.0,
                sensitivity=1.0,
                zeros=np.array([]),
                poles=np.array([]),
                unit='M/S'
            )
            
            # Compute energy and theta
            result = compute_seismic_energy(
                data=data,
                dt=dt,
                instrument=instrument,
                elat=EVENT['lat'], elon=EVENT['lon'],
                slat=sta_lat, slon=sta_lon,
                depth_km=EVENT['depth_km'],
                moment=EVENT['moment'],
                strike=EVENT['strike'], 
                dip=EVENT['dip'], 
                rake=EVENT['rake'],
                tmin=0.5, tmax=10.0,
                station_code=sta_code,
                jb_tables=jb_tables
            )
            
            results.append({
                'station': sta_code,
                'distance': dist_deg,
                'azimuth': az,
                'slowness': slowness,
                'energy': result.energy_estimated,
                'theta': result.theta_estimated
            })
            
            print(f"{sta_code:<15} {dist_deg:>8.2f} {az:>8.1f} {slowness:>10.4f} {result.energy_estimated:>12.2e} {result.theta_estimated:>8.2f}")
            
        except Exception as e:
            print(f"{sta_code:<15} -- Error: {str(e)[:35]}")
            continue

# Summary
print("="*70)

if results:
    theta_values = [r['theta'] for r in results]
    theta_mean = np.mean(theta_values)
    theta_std = np.std(theta_values)
    
    energy_values = [r['energy'] for r in results]
    energy_mean = np.mean(energy_values)
    
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Event: {EVENT['id']} {EVENT['name']}")
    print(f"  Mw = {EVENT['mw']}, Depth = {EVENT['depth_km']} km")
    print(f"")
    print(f"  Stations processed: {len(results)}")
    print(f"  Mean Energy: {energy_mean:.2e} J")
    print(f"")
    print(f"  ╔════════════════════════════════════╗")
    print(f"  ║   THETA (Θ) = {theta_mean:6.2f} ± {theta_std:5.2f}      ║")
    print(f"  ╚════════════════════════════════════╝")
    print(f"")
    print(f"  Individual values: {[f'{t:.2f}' for t in theta_values]}")
    print(f"{'='*70}")
    
    # Classification
    if theta_mean < -5.7:
        classification = "SLOW earthquake (possible tsunami)"
    elif theta_mean > -4.3:
        classification = "FAST rupture (high stress drop)"
    else:
        classification = "NORMAL earthquake"
    
    print(f"\n  Classification: {classification}")
    print(f"{'='*70}")
    
    # Save to file
    with open(f"./theta_{EVENT['id']}.txt", 'w') as f:
        f.write(f"Event: {EVENT['id']} - {EVENT['name']}\n")
        f.write(f"Date: {EVENT['date']} {EVENT['time']} UTC\n")
        f.write(f"Location: ({EVENT['lat']}, {EVENT['lon']}), Depth: {EVENT['depth_km']} km\n")
        f.write(f"Mw: {EVENT['mw']}, M0: {EVENT['moment']:.2e} dyn-cm\n")
        f.write(f"Mechanism: strike={EVENT['strike']}, dip={EVENT['dip']}, rake={EVENT['rake']}\n\n")
        f.write(f"Stations: {len(results)}\n")
        f.write(f"THETA = {theta_mean:.2f} ± {theta_std:.2f}\n\n")
        for r in results:
            f.write(f"{r['station']:<15} {r['distance']:>8.2f} {r['azimuth']:>8.1f} {r['theta']:>8.2f}\n")
    
    print(f"\nResults saved to: theta_{EVENT['id']}.txt")
    
    print(f"\n{'='*70}")
    print("COMPARE WITH YOUR FORTRAN RESULT:")
    print(f"{'='*70}")
    print(f"  Python  Θ = {theta_mean:.2f}")
    print(f"  Fortran Θ = ??? (what did you get?)")
    print(f"{'='*70}")
    
else:
    print("No stations processed successfully!")
