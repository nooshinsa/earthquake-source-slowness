#!/usr/bin/env python3
"""
DEBUG: Find out why stations are failing for Santa Cruz event
"""

import os
import numpy as np
from datetime import datetime
import traceback

# Event parameters
EVENT = {
    'id': '201302060112A',
    'lat': -11.18,
    'lon': 165.21,
    'depth_km': 20.2,
    'moment': 9.37e27,
    'mw': 7.9,
    'strike': 320,
    'dip': 20,
    'rake': 89
}

print("="*70)
print("DEBUG: Santa Cruz Islands Event")
print("="*70)

# Check obspy
try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
    print("✓ obspy imported successfully")
except ImportError as e:
    print(f"✗ obspy import error: {e}")
    exit(1)

# Connect to IRIS
print("\nConnecting to IRIS...")
try:
    client = Client("IRIS")
    print("✓ Connected to IRIS")
except Exception as e:
    print(f"✗ IRIS connection error: {e}")
    exit(1)

# Time window
origin = UTCDateTime("2013-02-06T01:12:55")
starttime = origin - 60
endtime = origin + 1800

print(f"\nTime window: {starttime} to {endtime}")

# Get stations
print("\nSearching for stations...")
try:
    inventory = client.get_stations(
        network="II,IU",
        channel="BHZ",
        starttime=starttime,
        endtime=endtime,
        latitude=EVENT['lat'],
        longitude=EVENT['lon'],
        minradius=30.0,
        maxradius=80.0,
        level="response"
    )
    print(f"✓ Got inventory")
except Exception as e:
    print(f"✗ Inventory error: {e}")
    traceback.print_exc()
    exit(1)

# List stations
print("\nStations found:")
station_list = []
for network in inventory:
    for station in network:
        sta_lat = station.latitude
        sta_lon = station.longitude
        dist_m, az, baz = gps2dist_azimuth(EVENT['lat'], EVENT['lon'], sta_lat, sta_lon)
        dist_deg = kilometers2degrees(dist_m / 1000.0)
        station_list.append({
            'net': network.code,
            'sta': station.code,
            'lat': sta_lat,
            'lon': sta_lon,
            'dist': dist_deg
        })
        print(f"  {network.code}.{station.code}: dist={dist_deg:.1f}°")

print(f"\nTotal: {len(station_list)} stations")

if len(station_list) == 0:
    print("\n✗ No stations found in range!")
    print("  Trying without distance filter...")
    
    inventory = client.get_stations(
        network="II,IU",
        channel="BHZ",
        starttime=starttime,
        endtime=endtime,
        level="station"
    )
    
    for network in inventory:
        for station in network:
            print(f"  {network.code}.{station.code}")
    exit(1)

# Try downloading ONE station
print("\n" + "="*70)
print("TESTING FIRST STATION DOWNLOAD:")
print("="*70)

test_sta = station_list[0]
print(f"\nTrying: {test_sta['net']}.{test_sta['sta']}")

# Step 1: Download waveform
print("\n1. Downloading waveform...")
try:
    st = client.get_waveforms(
        network=test_sta['net'],
        station=test_sta['sta'],
        location="*",
        channel="BHZ",
        starttime=starttime,
        endtime=endtime
    )
    print(f"   ✓ Got {len(st)} trace(s)")
    for tr in st:
        print(f"     {tr.id}: {tr.stats.npts} samples, {tr.stats.sampling_rate} Hz")
except Exception as e:
    print(f"   ✗ Download error: {e}")
    traceback.print_exc()
    exit(1)

if len(st) == 0:
    print("   ✗ No traces returned!")
    exit(1)

# Step 2: Merge if multiple traces
print("\n2. Merging traces...")
try:
    st.merge(fill_value=0)
    print(f"   ✓ Merged: {len(st)} trace(s)")
except Exception as e:
    print(f"   ✗ Merge error: {e}")

# Step 3: Remove response
print("\n3. Removing instrument response...")
try:
    st_copy = st.copy()
    st_copy.remove_response(inventory=inventory, output="VEL")
    print(f"   ✓ Response removed")
    tr = st_copy[0]
    print(f"     Data range: {tr.data.min():.2e} to {tr.data.max():.2e}")
except Exception as e:
    print(f"   ✗ Response removal error: {e}")
    traceback.print_exc()
    
    # Try without pre-filter
    print("\n   Trying without pre-filter...")
    try:
        st_copy = st.copy()
        st_copy.remove_response(inventory=inventory, output="VEL", pre_filt=None)
        print(f"   ✓ Response removed (no pre-filter)")
    except Exception as e2:
        print(f"   ✗ Still failing: {e2}")

# Step 4: Check our modules
print("\n4. Testing our modules...")
try:
    from depth_bins import get_depth_bin
    bin_info = get_depth_bin(EVENT['depth_km'])
    print(f"   ✓ depth_bins: {bin_info['name']}")
except Exception as e:
    print(f"   ✗ depth_bins error: {e}")
    traceback.print_exc()

try:
    from travel_time import JBTables
    jb = JBTables()
    slowness = jb.get_slowness(test_sta['dist'], EVENT['depth_km'])
    print(f"   ✓ JB tables: slowness = {slowness:.4f} s/deg")
except Exception as e:
    print(f"   ✗ JB tables error: {e}")
    traceback.print_exc()

try:
    from instrument_response import InstrumentResponse
    print(f"   ✓ instrument_response imported")
except Exception as e:
    print(f"   ✗ instrument_response error: {e}")
    traceback.print_exc()

try:
    from energy_calculation import compute_seismic_energy
    print(f"   ✓ energy_calculation imported")
except Exception as e:
    print(f"   ✗ energy_calculation error: {e}")
    traceback.print_exc()

# Step 5: Try energy calculation
print("\n5. Testing energy calculation...")
try:
    from instrument_response import InstrumentResponse
    from energy_calculation import compute_seismic_energy
    from travel_time import JBTables
    
    jb_tables = JBTables()
    
    # Get station coords from inventory
    for net in inventory:
        for sta in net:
            if sta.code == test_sta['sta']:
                sta_lat = sta.latitude
                sta_lon = sta.longitude
                break
    
    # Use the response-removed data
    tr = st_copy[0]
    data = tr.data.astype(float)
    dt = tr.stats.delta
    
    print(f"   Data: {len(data)} samples, dt={dt}s")
    print(f"   Data range: {data.min():.2e} to {data.max():.2e}")
    
    # Flat response (already removed)
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
        elat=EVENT['lat'], elon=EVENT['lon'],
        slat=sta_lat, slon=sta_lon,
        depth_km=EVENT['depth_km'],
        moment=EVENT['moment'],
        strike=EVENT['strike'], 
        dip=EVENT['dip'], 
        rake=EVENT['rake'],
        tmin=0.5, tmax=10.0,
        station_code=f"{test_sta['net']}.{test_sta['sta']}",
        jb_tables=jb_tables
    )
    
    print(f"   ✓ Energy calculation successful!")
    print(f"     Energy: {result.energy_joules:.2e} J")
    print(f"     Theta:  {result.theta_estimated:.2f}")
    
except Exception as e:
    print(f"   ✗ Energy calculation error: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("DEBUG COMPLETE")
print("="*70)



