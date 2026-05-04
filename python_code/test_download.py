#!/usr/bin/env python3
"""
Simple test to check if IRIS download works.
Run this in your seismo_env.
"""

print("Testing IRIS download...")
print("="*50)

# Step 1: Check obspy
print("\n1. Checking obspy installation...")
try:
    import obspy
    print(f"   ✓ obspy version: {obspy.__version__}")
except ImportError as e:
    print(f"   ✗ obspy NOT installed: {e}")
    print("\n   Install with: pip install obspy")
    exit(1)

# Step 2: Check FDSN client
print("\n2. Testing IRIS connection...")
try:
    from obspy.clients.fdsn import Client
    client = Client("IRIS")
    print("   ✓ Connected to IRIS")
except Exception as e:
    print(f"   ✗ Cannot connect to IRIS: {e}")
    print("\n   Check your internet connection")
    exit(1)

# Step 3: Try to get station inventory
print("\n3. Testing station inventory request...")
try:
    from obspy import UTCDateTime
    
    inv = client.get_stations(
        network="IU",
        station="ANMO",
        channel="BHZ",
        starttime=UTCDateTime("2023-01-01"),
        endtime=UTCDateTime("2023-01-02"),
        level="station"
    )
    print(f"   ✓ Got inventory: {len(inv)} networks")
    for net in inv:
        for sta in net:
            print(f"      {net.code}.{sta.code}: ({sta.latitude}, {sta.longitude})")
except Exception as e:
    print(f"   ✗ Inventory request failed: {e}")
    exit(1)

# Step 4: Try to download a small waveform
print("\n4. Testing waveform download...")
try:
    st = client.get_waveforms(
        network="IU",
        station="ANMO",
        location="00",
        channel="BHZ",
        starttime=UTCDateTime("2023-01-01T00:00:00"),
        endtime=UTCDateTime("2023-01-01T00:01:00")  # Just 1 minute
    )
    print(f"   ✓ Downloaded waveform: {len(st)} traces")
    print(f"      {st[0].stats.npts} samples at {st[0].stats.sampling_rate} Hz")
except Exception as e:
    print(f"   ✗ Waveform download failed: {e}")
    exit(1)

print("\n" + "="*50)
print("ALL TESTS PASSED! IRIS download is working.")
print("="*50)
print("\nNow try running the full pipeline:")
print("  python3 iris_downloader.py --start 2023-01-01 --end 2023-01-31 --min-mag 7.0")



