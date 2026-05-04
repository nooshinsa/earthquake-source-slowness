#!/usr/bin/env python3
"""
IRIS Data Downloader and Processor

Automatically downloads seismic data from IRIS for events in a CMT catalog
and computes Theta (Θ) for each event.

Features:
    - Automatically fetch CMT catalog from Global CMT or USGS
    - Download BHZ waveforms from IRIS
    - Download instrument responses
    - Compute Theta for all events

Requirements:
    pip install obspy numpy

Usage:
    # Fetch CMT events and process automatically:
    python iris_downloader.py --start 2023-01-01 --end 2023-12-31 --min-mag 6.5
    
    # Or use existing catalog:
    python iris_downloader.py --catalog events.csv --output results/
    
    # Demo:
    python iris_downloader.py --demo
"""

import os
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
import urllib.request
import json

# Check for obspy
try:
    from obspy import UTCDateTime, read, read_inventory
    from obspy.clients.fdsn import Client
    from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    print("WARNING: obspy not installed. Install with: pip install obspy")


@dataclass
class CMTEvent:
    """
    CMT event parameters.
    """
    event_id: str
    origin_time: datetime
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float
    moment: float  # in dyn-cm
    strike: float
    dip: float
    rake: float


# =============================================================================
# AUTOMATIC CMT CATALOG FETCHING
# =============================================================================

def fetch_cmt_catalog(
    start_date: str,
    end_date: str,
    min_magnitude: float = 6.0,
    max_magnitude: float = 10.0,
    min_depth: float = 0.0,
    max_depth: float = 700.0,
    min_latitude: float = -90.0,
    max_latitude: float = 90.0,
    min_longitude: float = -180.0,
    max_longitude: float = 180.0,
    source: str = "USGS"
) -> List[CMTEvent]:
    """
    Automatically fetch CMT events from online catalogs.
    
    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    min_magnitude, max_magnitude : float
        Magnitude range
    min_depth, max_depth : float
        Depth range in km
    min_latitude, max_latitude : float
        Latitude range
    min_longitude, max_longitude : float
        Longitude range
    source : str
        Data source: "USGS", "GCMT", or "ISC"
        
    Returns
    -------
    list
        List of CMTEvent objects
    """
    print(f"\n{'='*70}")
    print(f"FETCHING CMT CATALOG")
    print(f"{'='*70}")
    print(f"Source: {source}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Magnitude: {min_magnitude} - {max_magnitude}")
    print(f"Depth: {min_depth} - {max_depth} km")
    
    if source.upper() == "USGS":
        events = fetch_from_usgs(
            start_date, end_date, min_magnitude, max_magnitude,
            min_depth, max_depth, min_latitude, max_latitude,
            min_longitude, max_longitude
        )
    elif source.upper() == "GCMT":
        events = fetch_from_globalcmt(
            start_date, end_date, min_magnitude, max_magnitude,
            min_depth, max_depth
        )
    elif source.upper() == "ISC":
        events = fetch_from_isc(
            start_date, end_date, min_magnitude, max_magnitude,
            min_depth, max_depth
        )
    else:
        print(f"Unknown source: {source}. Using USGS.")
        events = fetch_from_usgs(
            start_date, end_date, min_magnitude, max_magnitude,
            min_depth, max_depth, min_latitude, max_latitude,
            min_longitude, max_longitude
        )
    
    print(f"\nFound {len(events)} events")
    print(f"{'='*70}\n")
    
    return events


def fetch_from_usgs(
    start_date: str,
    end_date: str,
    min_mag: float,
    max_mag: float,
    min_depth: float,
    max_depth: float,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
) -> List[CMTEvent]:
    """
    Fetch events from USGS Earthquake Catalog API.
    
    This provides moment tensor solutions for larger earthquakes.
    """
    events = []
    
    # USGS API URL
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": min_mag,
        "maxmagnitude": max_mag,
        "mindepth": min_depth,
        "maxdepth": max_depth,
        "minlatitude": min_lat,
        "maxlatitude": max_lat,
        "minlongitude": min_lon,
        "maxlongitude": max_lon,
        "producttype": "moment-tensor",  # Only events with moment tensors
        "orderby": "time"
    }
    
    # Build URL
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{base_url}?{query_string}"
    
    print(f"Fetching from USGS...")
    
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode())
        
        features = data.get("features", [])
        print(f"  Retrieved {len(features)} events with moment tensors")
        
        for feature in features:
            props = feature["properties"]
            geom = feature["geometry"]
            
            # Extract basic info
            event_id = props.get("code", props.get("ids", "unknown").split(",")[0])
            
            # Time
            time_ms = props.get("time", 0)
            origin_time = datetime.utcfromtimestamp(time_ms / 1000.0)
            
            # Location
            coords = geom.get("coordinates", [0, 0, 0])
            lon = coords[0]
            lat = coords[1]
            depth = coords[2]
            
            # Magnitude
            mag = props.get("mag", 0)
            
            # Try to get moment tensor info
            moment = magnitude_to_moment(mag)
            strike, dip, rake = 0.0, 45.0, 90.0  # Defaults
            
            # Check for moment tensor product
            if "products" in props:
                mt_products = props.get("products", {}).get("moment-tensor", [])
                if mt_products:
                    mt = mt_products[0].get("properties", {})
                    
                    # Get scalar moment if available
                    if "scalar-moment" in mt:
                        moment = float(mt["scalar-moment"]) * 1e7  # Convert to dyn-cm
                    
                    # Get nodal plane 1
                    if "nodal-plane-1-strike" in mt:
                        strike = float(mt.get("nodal-plane-1-strike", 0))
                        dip = float(mt.get("nodal-plane-1-dip", 45))
                        rake = float(mt.get("nodal-plane-1-rake", 90))
            
            events.append(CMTEvent(
                event_id=event_id,
                origin_time=origin_time,
                latitude=lat,
                longitude=lon,
                depth_km=depth,
                magnitude=mag,
                moment=moment,
                strike=strike,
                dip=dip,
                rake=rake
            ))
            
    except Exception as e:
        print(f"  Error fetching from USGS: {e}")
    
    return events


def fetch_from_globalcmt(
    start_date: str,
    end_date: str,
    min_mag: float,
    max_mag: float,
    min_depth: float,
    max_depth: float
) -> List[CMTEvent]:
    """
    Fetch events from Global CMT catalog.
    
    Uses the NDK format files from globalcmt.org
    """
    events = []
    
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"Fetching from Global CMT...")
    
    # Global CMT provides monthly NDK files
    # URL format: https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/YYYY/MMMYY.ndk
    
    current = start
    while current <= end:
        year = current.year
        month = current.month
        month_name = current.strftime("%b").lower()
        year_short = current.strftime("%y")
        
        # Try different URL patterns
        urls = [
            f"https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/{year}/{month_name}{year_short}.ndk",
            f"https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_QUICK/qcmt.ndk"
        ]
        
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    ndk_data = response.read().decode('utf-8', errors='ignore')
                
                # Parse NDK format
                month_events = parse_ndk_string(ndk_data)
                
                # Filter by date and magnitude
                for ev in month_events:
                    if start <= ev.origin_time <= end:
                        if min_mag <= ev.magnitude <= max_mag:
                            if min_depth <= ev.depth_km <= max_depth:
                                events.append(ev)
                
                print(f"  {current.strftime('%Y-%m')}: {len([e for e in month_events if start <= e.origin_time <= end])} events")
                break
                
            except Exception as e:
                continue
        
        # Move to next month
        if month == 12:
            current = datetime(year + 1, 1, 1)
        else:
            current = datetime(year, month + 1, 1)
    
    return events


def fetch_from_isc(
    start_date: str,
    end_date: str,
    min_mag: float,
    max_mag: float,
    min_depth: float,
    max_depth: float
) -> List[CMTEvent]:
    """
    Fetch events from ISC (International Seismological Centre).
    """
    events = []
    
    # ISC FDSN web service
    base_url = "http://www.isc.ac.uk/fdsnws/event/1/query"
    
    params = {
        "format": "text",
        "starttime": start_date,
        "endtime": end_date,
        "minmag": min_mag,
        "maxmag": max_mag,
        "mindepth": min_depth,
        "maxdepth": max_depth
    }
    
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{base_url}?{query_string}"
    
    print(f"Fetching from ISC...")
    
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            lines = response.read().decode().split('\n')
        
        # Skip header
        for line in lines[1:]:
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) < 10:
                continue
            
            try:
                event_id = parts[0].strip()
                time_str = parts[1].strip()
                lat = float(parts[2])
                lon = float(parts[3])
                depth = float(parts[4]) if parts[4].strip() else 10.0
                mag = float(parts[10]) if len(parts) > 10 and parts[10].strip() else 0
                
                origin_time = datetime.fromisoformat(time_str.replace('Z', ''))
                moment = magnitude_to_moment(mag)
                
                events.append(CMTEvent(
                    event_id=event_id,
                    origin_time=origin_time,
                    latitude=lat,
                    longitude=lon,
                    depth_km=depth,
                    magnitude=mag,
                    moment=moment,
                    strike=0.0,
                    dip=45.0,
                    rake=90.0
                ))
            except:
                continue
        
        print(f"  Retrieved {len(events)} events")
        
    except Exception as e:
        print(f"  Error fetching from ISC: {e}")
    
    return events


def parse_ndk_string(ndk_string: str) -> List[CMTEvent]:
    """
    Parse NDK format string into CMTEvent objects.
    """
    events = []
    lines = ndk_string.strip().split('\n')
    
    # Process 5 lines at a time
    i = 0
    while i + 4 < len(lines):
        try:
            line1 = lines[i]
            line2 = lines[i+1]
            line3 = lines[i+2]
            line4 = lines[i+3]
            line5 = lines[i+4]
            
            # Line 1: Hypocenter
            parts1 = line1.split()
            if len(parts1) < 10:
                i += 5
                continue
            
            # Date and time
            year = int(parts1[1])
            month = int(parts1[2])
            day = int(parts1[3])
            hour = int(parts1[4])
            minute = int(parts1[5])
            second = float(parts1[6])
            
            lat = float(parts1[7])
            lon = float(parts1[8])
            depth = float(parts1[9])
            
            origin_time = datetime(year, month, day, hour, minute, int(second))
            
            # Line 2: CMT info
            parts2 = line2.split()
            event_id = parts2[0] if parts2 else f"{year}{month:02d}{day:02d}"
            
            # Line 4: Moment tensor
            parts4 = line4.split()
            
            # Get exponent and moment tensor components
            try:
                exp = int(parts4[0])
                # Scalar moment from Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
                mrr = float(parts4[1])
                mtt = float(parts4[3])
                mpp = float(parts4[5])
                
                # Approximate scalar moment
                moment = np.sqrt(0.5 * (mrr**2 + mtt**2 + mpp**2)) * (10 ** exp) * 1e7
            except:
                moment = 1e27
            
            # Calculate magnitude from moment
            mag = (np.log10(moment) - 16.1) / 1.5
            
            # Line 5: Principal axes (contains strike/dip/rake for planes)
            # Format varies, use defaults for now
            strike, dip, rake = 0.0, 45.0, 90.0
            
            events.append(CMTEvent(
                event_id=event_id,
                origin_time=origin_time,
                latitude=lat,
                longitude=lon,
                depth_km=depth,
                magnitude=mag,
                moment=moment,
                strike=strike,
                dip=dip,
                rake=rake
            ))
            
        except Exception as e:
            pass
        
        i += 5
    
    return events


def magnitude_to_moment(magnitude: float) -> float:
    """
    Convert moment magnitude to seismic moment (dyn-cm).
    
    M0 = 10^(1.5*Mw + 16.1) dyn-cm
    """
    return 10 ** (1.5 * magnitude + 16.1)


def save_catalog_to_csv(events: List[CMTEvent], filename: str):
    """
    Save CMT events to CSV file.
    """
    with open(filename, 'w') as f:
        f.write("event_id,datetime,lat,lon,depth,mag,moment,strike,dip,rake\n")
        for ev in events:
            f.write(f"{ev.event_id},{ev.origin_time.isoformat()},"
                    f"{ev.latitude},{ev.longitude},{ev.depth_km},"
                    f"{ev.magnitude:.2f},{ev.moment:.3e},"
                    f"{ev.strike},{ev.dip},{ev.rake}\n")
    print(f"Catalog saved to: {filename}")
    

def read_cmt_catalog(filename: str) -> List[CMTEvent]:
    """
    Read CMT catalog from CSV or NDK format.
    
    CSV format expected columns:
        event_id, datetime, lat, lon, depth, mag, moment, strike, dip, rake
    
    Parameters
    ----------
    filename : str
        Path to catalog file
        
    Returns
    -------
    list
        List of CMTEvent objects
    """
    events = []
    
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.csv':
        events = read_csv_catalog(filename)
    elif ext == '.ndk':
        events = read_ndk_catalog(filename)
    else:
        # Try CSV format
        events = read_csv_catalog(filename)
    
    return events


def read_csv_catalog(filename: str) -> List[CMTEvent]:
    """
    Read CSV format catalog.
    
    Expected columns (comma or tab separated):
    event_id, datetime, lat, lon, depth, mag, moment, strike, dip, rake
    
    datetime format: YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM:SS
    """
    events = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header if present
    start_line = 0
    if lines[0].lower().startswith('event') or lines[0].lower().startswith('#'):
        start_line = 1
    
    for line in lines[start_line:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Split by comma or tab
        if ',' in line:
            parts = [p.strip() for p in line.split(',')]
        else:
            parts = line.split()
        
        if len(parts) < 10:
            print(f"Warning: Skipping line (not enough columns): {line[:50]}...")
            continue
        
        try:
            event_id = parts[0]
            
            # Parse datetime
            dt_str = parts[1]
            if 'T' in dt_str:
                origin_time = datetime.fromisoformat(dt_str.replace('Z', ''))
            else:
                # Try common formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y%m%d%H%M%S']:
                    try:
                        origin_time = datetime.strptime(dt_str, fmt)
                        break
                    except:
                        continue
            
            lat = float(parts[2])
            lon = float(parts[3])
            depth = float(parts[4])
            mag = float(parts[5])
            moment = float(parts[6])
            strike = float(parts[7])
            dip = float(parts[8])
            rake = float(parts[9])
            
            events.append(CMTEvent(
                event_id=event_id,
                origin_time=origin_time,
                latitude=lat,
                longitude=lon,
                depth_km=depth,
                magnitude=mag,
                moment=moment,
                strike=strike,
                dip=dip,
                rake=rake
            ))
            
        except Exception as e:
            print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
            continue
    
    return events


def read_ndk_catalog(filename: str) -> List[CMTEvent]:
    """
    Read Global CMT NDK format catalog.
    
    NDK format has 5 lines per event.
    """
    events = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Process 5 lines at a time
    for i in range(0, len(lines), 5):
        if i + 4 >= len(lines):
            break
        
        try:
            line1 = lines[i].strip()
            line2 = lines[i+1].strip()
            line3 = lines[i+2].strip()
            line4 = lines[i+3].strip()
            line5 = lines[i+4].strip()
            
            # Line 1: Hypocenter info
            # PDE  2023  1 15  3 45 12.3  -15.23  -75.12  33.0 5.8 5.9 PERU
            parts1 = line1.split()
            year = int(parts1[1])
            month = int(parts1[2])
            day = int(parts1[3])
            hour = int(parts1[4])
            minute = int(parts1[5])
            second = float(parts1[6])
            lat = float(parts1[7])
            lon = float(parts1[8])
            depth = float(parts1[9])
            
            origin_time = datetime(year, month, day, hour, minute, int(second))
            
            # Line 2: CMT info
            event_id = line2.split()[0]
            
            # Line 4: Centroid info and moment tensor
            parts4 = line4.split()
            
            # Line 5: Principal axes and moment
            parts5 = line5.split()
            # Scalar moment is usually the last value in scientific notation
            moment_str = [p for p in parts5 if 'E' in p.upper()][-1]
            moment = float(moment_str) * 1e7  # Convert to dyn-cm
            
            # Get magnitude from moment
            mag = (np.log10(moment) - 16.1) / 1.5
            
            # Focal mechanism - would need more parsing
            # For now use defaults
            strike, dip, rake = 0.0, 45.0, 90.0
            
            events.append(CMTEvent(
                event_id=event_id,
                origin_time=origin_time,
                latitude=lat,
                longitude=lon,
                depth_km=depth,
                magnitude=mag,
                moment=moment,
                strike=strike,
                dip=dip,
                rake=rake
            ))
            
        except Exception as e:
            print(f"Warning: Could not parse NDK event at line {i}: {e}")
            continue
    
    return events


def download_event_data(
    event: CMTEvent,
    output_dir: str,
    networks: str = "II,IU",
    channel: str = "BHZ",
    min_distance: float = 30.0,
    max_distance: float = 80.0,
    pre_origin: float = 60.0,  # seconds before origin
    post_origin: float = 1800.0,  # seconds after origin (30 min)
    client_name: str = "IRIS"
) -> List[str]:
    """
    Download waveform data and responses from IRIS for an event.
    
    Parameters
    ----------
    event : CMTEvent
        Event to download data for
    output_dir : str
        Directory to save data
    networks : str
        Comma-separated network codes (default: "II,IU")
    channel : str
        Channel code (default: "BHZ")
    min_distance, max_distance : float
        Distance range in degrees
    pre_origin, post_origin : float
        Time window relative to origin (seconds)
    client_name : str
        FDSN client name
        
    Returns
    -------
    list
        List of downloaded file paths
    """
    if not OBSPY_AVAILABLE:
        raise ImportError("obspy is required for downloading data")
    
    # Create output directory
    event_dir = os.path.join(output_dir, event.event_id.replace(' ', '_'))
    os.makedirs(event_dir, exist_ok=True)
    
    # Initialize FDSN client
    client = Client(client_name)
    
    # Time window
    starttime = UTCDateTime(event.origin_time) - pre_origin
    endtime = UTCDateTime(event.origin_time) + post_origin
    
    downloaded_files = []
    
    print(f"\nDownloading data for {event.event_id}...")
    print(f"  Origin: {event.origin_time}")
    print(f"  Location: ({event.latitude}, {event.longitude}), depth={event.depth_km} km")
    print(f"  Networks: {networks}, Channel: {channel}")
    print(f"  Distance range: {min_distance}° - {max_distance}°")
    
    # Get station inventory within distance range
    try:
        inventory = client.get_stations(
            network=networks,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
            latitude=event.latitude,
            longitude=event.longitude,
            minradius=min_distance,
            maxradius=max_distance,
            level="response"
        )
    except Exception as e:
        print(f"  Error getting station inventory: {e}")
        return []
    
    n_stations = sum(len(net) for net in inventory)
    print(f"  Found {n_stations} stations in distance range")
    
    # Download waveforms for each station
    for network in inventory:
        for station in network:
            sta_code = f"{network.code}.{station.code}"
            
            try:
                # Get waveforms
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
                
                # Get station coordinates
                sta_lat = station.latitude
                sta_lon = station.longitude
                
                # Calculate distance
                dist_m, az, baz = gps2dist_azimuth(
                    event.latitude, event.longitude,
                    sta_lat, sta_lon
                )
                dist_deg = kilometers2degrees(dist_m / 1000.0)
                
                # Save to miniSEED
                mseed_file = os.path.join(event_dir, f"{sta_code}.{channel}.mseed")
                st.write(mseed_file, format="MSEED")
                
                # Save response (StationXML)
                sta_inv = inventory.select(network=network.code, station=station.code)
                resp_file = os.path.join(event_dir, f"{sta_code}.{channel}.xml")
                sta_inv.write(resp_file, format="STATIONXML")
                
                # Save metadata
                meta_file = os.path.join(event_dir, f"{sta_code}.{channel}.meta")
                with open(meta_file, 'w') as f:
                    f.write(f"station_lat={sta_lat}\n")
                    f.write(f"station_lon={sta_lon}\n")
                    f.write(f"distance_deg={dist_deg}\n")
                    f.write(f"azimuth={az}\n")
                    f.write(f"back_azimuth={baz}\n")
                
                downloaded_files.append(mseed_file)
                print(f"    ✓ {sta_code}: {dist_deg:.1f}°, az={az:.1f}°")
                
            except Exception as e:
                print(f"    ✗ {sta_code}: {e}")
                continue
    
    print(f"  Downloaded {len(downloaded_files)} stations")
    
    # Save event info
    event_file = os.path.join(event_dir, "event_info.txt")
    with open(event_file, 'w') as f:
        f.write(f"event_id={event.event_id}\n")
        f.write(f"origin_time={event.origin_time.isoformat()}\n")
        f.write(f"latitude={event.latitude}\n")
        f.write(f"longitude={event.longitude}\n")
        f.write(f"depth_km={event.depth_km}\n")
        f.write(f"magnitude={event.magnitude}\n")
        f.write(f"moment={event.moment}\n")
        f.write(f"strike={event.strike}\n")
        f.write(f"dip={event.dip}\n")
        f.write(f"rake={event.rake}\n")
    
    return downloaded_files


def process_downloaded_event(
    event_dir: str,
    event: CMTEvent
) -> dict:
    """
    Process downloaded data for an event and compute Theta.
    
    Parameters
    ----------
    event_dir : str
        Directory containing downloaded data
    event : CMTEvent
        Event parameters
        
    Returns
    -------
    dict
        Results dictionary
    """
    from obspy import read, read_inventory
    from energy_calculation import compute_seismic_energy, classify_event
    from instrument_response import InstrumentResponse
    from travel_time import JBTables
    from depth_bins import get_depth_bin
    
    print(f"\nProcessing {event.event_id}...")
    
    # Initialize JB tables
    jb_tables = JBTables()
    
    # Find all mseed files
    mseed_files = [f for f in os.listdir(event_dir) if f.endswith('.mseed')]
    
    results = []
    
    for mseed_file in mseed_files:
        sta_code = mseed_file.replace('.mseed', '').rsplit('.', 1)[0]
        
        try:
            # Read waveform
            mseed_path = os.path.join(event_dir, mseed_file)
            st = read(mseed_path)
            tr = st[0]
            
            data = tr.data.astype(float)
            dt = tr.stats.delta
            
            # Read response
            xml_file = mseed_file.replace('.mseed', '.xml')
            xml_path = os.path.join(event_dir, xml_file)
            
            if os.path.exists(xml_path):
                inv = read_inventory(xml_path)
                # Remove response to get velocity
                st_vel = st.copy()
                st_vel.remove_response(inventory=inv, output="VEL")
                data = st_vel[0].data.astype(float)
                
                # Create instrument response object (flat since we already removed it)
                instrument = InstrumentResponse(
                    a0=1.0,
                    sensitivity=1.0,
                    zeros=np.array([]),
                    poles=np.array([]),
                    unit='M/S'
                )
            else:
                # Use default response
                instrument = InstrumentResponse(
                    a0=5.714e8,
                    sensitivity=629.0,
                    zeros=np.array([0+0j, 0+0j]),
                    poles=np.array([
                        -0.01234+0.01234j, -0.01234-0.01234j,
                        -39.18+49.12j, -39.18-49.12j
                    ]),
                    unit='M/S'
                )
            
            # Read metadata
            meta_file = mseed_file.replace('.mseed', '.meta')
            meta_path = os.path.join(event_dir, meta_file)
            
            sta_lat, sta_lon = 0.0, 0.0
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    for line in f:
                        if line.startswith('station_lat='):
                            sta_lat = float(line.split('=')[1])
                        elif line.startswith('station_lon='):
                            sta_lon = float(line.split('=')[1])
            
            # Compute energy
            result = compute_seismic_energy(
                data=data,
                dt=dt,
                instrument=instrument,
                elat=event.latitude,
                elon=event.longitude,
                slat=sta_lat,
                slon=sta_lon,
                depth_km=event.depth_km,
                moment=event.moment,
                strike=event.strike,
                dip=event.dip,
                rake=event.rake,
                tmin=0.5,
                tmax=10.0,
                station_code=sta_code,
                jb_tables=jb_tables
            )
            
            results.append(result)
            print(f"  ✓ {sta_code}: Θ = {result.theta_estimated:.2f}")
            
        except Exception as e:
            print(f"  ✗ {sta_code}: {e}")
            continue
    
    # Compute average
    if results:
        theta_values = [r.theta_estimated for r in results]
        theta_mean = np.mean(theta_values)
        theta_std = np.std(theta_values)
        
        return {
            'event_id': event.event_id,
            'n_stations': len(results),
            'theta_mean': theta_mean,
            'theta_std': theta_std,
            'theta_values': theta_values,
            'classification': classify_event(theta_mean),
            'results': results
        }
    else:
        return {
            'event_id': event.event_id,
            'n_stations': 0,
            'theta_mean': np.nan,
            'theta_std': np.nan,
            'theta_values': [],
            'classification': 'NO DATA',
            'results': []
        }


def process_catalog(
    catalog_file: str,
    output_dir: str,
    networks: str = "II,IU",
    channel: str = "BHZ",
    min_distance: float = 30.0,
    max_distance: float = 80.0,
    download: bool = True
) -> List[dict]:
    """
    Process entire CMT catalog.
    
    Parameters
    ----------
    catalog_file : str
        Path to CMT catalog (CSV or NDK)
    output_dir : str
        Output directory for data and results
    networks, channel : str
        Network and channel codes
    min_distance, max_distance : float
        Distance range in degrees
    download : bool
        If True, download data. If False, only process existing data.
        
    Returns
    -------
    list
        List of result dictionaries for each event
    """
    # Read catalog
    print(f"\nReading catalog: {catalog_file}")
    events = read_cmt_catalog(catalog_file)
    print(f"Found {len(events)} events")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for i, event in enumerate(events):
        print(f"\n{'='*70}")
        print(f"Event {i+1}/{len(events)}: {event.event_id}")
        print(f"{'='*70}")
        
        event_dir = os.path.join(output_dir, event.event_id.replace(' ', '_'))
        
        # Download data if requested
        if download:
            try:
                download_event_data(
                    event=event,
                    output_dir=output_dir,
                    networks=networks,
                    channel=channel,
                    min_distance=min_distance,
                    max_distance=max_distance
                )
            except Exception as e:
                print(f"Error downloading: {e}")
                continue
        
        # Process data
        if os.path.exists(event_dir):
            result = process_downloaded_event(event_dir, event)
            all_results.append(result)
            
            print(f"\n  Summary: Θ = {result['theta_mean']:.2f} ± {result['theta_std']:.2f}")
            print(f"  Classification: {result['classification']}")
    
    # Write summary
    summary_file = os.path.join(output_dir, "theta_summary.csv")
    with open(summary_file, 'w') as f:
        f.write("event_id,n_stations,theta_mean,theta_std,classification\n")
        for r in all_results:
            f.write(f"{r['event_id']},{r['n_stations']},{r['theta_mean']:.3f},"
                    f"{r['theta_std']:.3f},{r['classification']}\n")
    
    print(f"\n{'='*70}")
    print(f"COMPLETE: Processed {len(all_results)} events")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*70}")
    
    return all_results


def create_sample_catalog(filename: str):
    """
    Create a sample CSV catalog file for testing.
    """
    sample = """# Sample CMT Catalog
# event_id, datetime, lat, lon, depth, mag, moment, strike, dip, rake
Peru_2023a,2023-03-15T12:30:45,-15.23,-75.12,33.0,7.2,1.5e27,320,45,90
Chile_2023a,2023-04-20T08:15:30,-22.45,-70.25,45.0,6.8,5.0e26,10,30,85
Japan_2023a,2023-05-10T14:22:18,38.5,142.1,28.0,7.0,8.0e26,195,15,95
Alaska_2023a,2023-06-05T03:45:12,55.2,-158.8,55.0,6.5,2.5e26,245,50,75
"""
    with open(filename, 'w') as f:
        f.write(sample)
    print(f"Sample catalog written to: {filename}")


def demo():
    """
    Demonstration of the IRIS downloader workflow.
    """
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║              IRIS DATA DOWNLOADER & THETA CALCULATOR                  ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Automatically downloads seismic data from IRIS and computes Theta    ║
╚═══════════════════════════════════════════════════════════════════════╝

This tool:
1. Reads your CMT catalog (event locations, depths, moments, mechanisms)
2. Downloads BHZ data from IRIS (II/IU networks, 30-80° distance)
3. Downloads instrument responses automatically
4. Computes Theta for each station
5. Averages across all stations for each event

CATALOG FORMAT (CSV):
─────────────────────────────────────────────────────────────────────────
event_id, datetime, lat, lon, depth, mag, moment, strike, dip, rake
Peru_2023, 2023-03-15T12:30:45, -15.23, -75.12, 33.0, 7.2, 1.5e27, 320, 45, 90
Chile_2023, 2023-04-20T08:15:30, -22.45, -70.25, 45.0, 6.8, 5.0e26, 10, 30, 85
─────────────────────────────────────────────────────────────────────────

USAGE:
─────────────────────────────────────────────────────────────────────────

1. Create your catalog file (see format above)

2. Run from Python:

   from iris_downloader import process_catalog
   
   results = process_catalog(
       catalog_file='my_events.csv',
       output_dir='./data/',
       networks='II,IU',
       channel='BHZ',
       min_distance=30.0,
       max_distance=80.0
   )

3. Or from command line:

   python iris_downloader.py --catalog my_events.csv --output ./data/

─────────────────────────────────────────────────────────────────────────
""")
    
    if not OBSPY_AVAILABLE:
        print("\n⚠ WARNING: obspy is not installed!")
        print("  Install with: pip install obspy")
        print("  Then you can download data from IRIS automatically.\n")
    else:
        print("\n✓ obspy is installed - ready to download from IRIS!\n")
    
    # Create sample catalog
    sample_file = "sample_catalog.csv"
    create_sample_catalog(sample_file)
    
    print(f"\nTo test with the sample catalog:")
    print(f"  python iris_downloader.py --catalog {sample_file} --output ./test_data/")


def full_automatic_pipeline(
    start_date: str,
    end_date: str,
    min_magnitude: float = 6.0,
    max_magnitude: float = 10.0,
    min_depth: float = 0.0,
    max_depth: float = 700.0,
    output_dir: str = "./theta_results/",
    networks: str = "II,IU",
    channel: str = "BHZ",
    min_distance: float = 30.0,
    max_distance: float = 80.0,
    source: str = "USGS"
) -> List[dict]:
    """
    Complete automatic pipeline:
    1. Fetch CMT catalog from online source
    2. Download waveforms from IRIS
    3. Compute Theta for all events
    
    Parameters
    ----------
    start_date, end_date : str
        Date range (YYYY-MM-DD format)
    min_magnitude, max_magnitude : float
        Magnitude range
    min_depth, max_depth : float
        Depth range in km
    output_dir : str
        Output directory
    networks : str
        Network codes
    channel : str
        Channel code
    min_distance, max_distance : float
        Station distance range in degrees
    source : str
        Catalog source: "USGS", "GCMT", or "ISC"
        
    Returns
    -------
    list
        Results for all events
    """
    print("\n" + "="*70)
    print("AUTOMATIC THETA CALCULATION PIPELINE")
    print("="*70)
    
    # Step 1: Fetch catalog
    events = fetch_cmt_catalog(
        start_date=start_date,
        end_date=end_date,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        min_depth=min_depth,
        max_depth=max_depth,
        source=source
    )
    
    if not events:
        print("No events found. Check your search parameters.")
        return []
    
    # Save catalog for reference
    os.makedirs(output_dir, exist_ok=True)
    catalog_file = os.path.join(output_dir, "fetched_catalog.csv")
    save_catalog_to_csv(events, catalog_file)
    
    # Step 2 & 3: Download and process each event
    all_results = []
    
    for i, event in enumerate(events):
        print(f"\n{'='*70}")
        print(f"EVENT {i+1}/{len(events)}: {event.event_id}")
        print(f"  {event.origin_time} | M{event.magnitude:.1f} | {event.depth_km:.0f} km")
        print(f"  ({event.latitude:.2f}, {event.longitude:.2f})")
        print(f"{'='*70}")
        
        event_dir = os.path.join(output_dir, event.event_id.replace(' ', '_').replace('/', '_'))
        
        # Download data
        try:
            downloaded = download_event_data(
                event=event,
                output_dir=output_dir,
                networks=networks,
                channel=channel,
                min_distance=min_distance,
                max_distance=max_distance
            )
            
            if not downloaded:
                print("  No data downloaded, skipping...")
                continue
                
        except Exception as e:
            print(f"  Download error: {e}")
            continue
        
        # Process data
        try:
            result = process_downloaded_event(event_dir, event)
            all_results.append(result)
            
            print(f"\n  ★ RESULT: Θ = {result['theta_mean']:.2f} ± {result['theta_std']:.2f}")
            print(f"    Classification: {result['classification']}")
            
        except Exception as e:
            print(f"  Processing error: {e}")
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Events processed: {len(all_results)}/{len(events)}")
    
    if all_results:
        # Save summary
        summary_file = os.path.join(output_dir, "theta_summary.csv")
        with open(summary_file, 'w') as f:
            f.write("event_id,datetime,lat,lon,depth,mag,n_stations,theta_mean,theta_std,classification\n")
            for r in all_results:
                ev = next((e for e in events if e.event_id == r['event_id']), None)
                if ev:
                    f.write(f"{r['event_id']},{ev.origin_time.isoformat()},"
                            f"{ev.latitude},{ev.longitude},{ev.depth_km},{ev.magnitude:.1f},"
                            f"{r['n_stations']},{r['theta_mean']:.3f},{r['theta_std']:.3f},"
                            f"{r['classification']}\n")
        
        print(f"\nResults saved to: {summary_file}")
        
        # Print table
        print(f"\n{'Event ID':<20} {'Mag':>5} {'Depth':>6} {'Θ':>7} {'±':>6} {'Class':<15}")
        print("-" * 70)
        for r in all_results:
            ev = next((e for e in events if e.event_id == r['event_id']), None)
            if ev:
                print(f"{r['event_id']:<20} {ev.magnitude:>5.1f} {ev.depth_km:>6.0f} "
                      f"{r['theta_mean']:>7.2f} {r['theta_std']:>6.2f} {r['classification']:<15}")
    
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download IRIS data and compute Theta for CMT events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Automatic: Fetch events and compute Theta
  python iris_downloader.py --start 2023-01-01 --end 2023-06-30 --min-mag 6.5
  
  # Use existing catalog file
  python iris_downloader.py --catalog my_events.csv
  
  # Demo mode
  python iris_downloader.py --demo
        """
    )
    
    # Automatic catalog fetching
    parser.add_argument('--start', type=str, 
                       help='Start date (YYYY-MM-DD) for automatic catalog fetch')
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD) for automatic catalog fetch')
    parser.add_argument('--min-mag', type=float, default=6.0,
                       help='Minimum magnitude (default: 6.0)')
    parser.add_argument('--max-mag', type=float, default=10.0,
                       help='Maximum magnitude (default: 10.0)')
    parser.add_argument('--min-depth', type=float, default=0.0,
                       help='Minimum depth in km (default: 0)')
    parser.add_argument('--max-depth', type=float, default=700.0,
                       help='Maximum depth in km (default: 700)')
    parser.add_argument('--source', type=str, default='USGS',
                       choices=['USGS', 'GCMT', 'ISC'],
                       help='Catalog source (default: USGS)')
    
    # Existing catalog
    parser.add_argument('--catalog', type=str, 
                       help='Path to existing CMT catalog file (CSV or NDK)')
    
    # Output and processing options
    parser.add_argument('--output', type=str, default='./theta_results/', 
                       help='Output directory (default: ./theta_results/)')
    parser.add_argument('--networks', type=str, default='II,IU',
                       help='Network codes (default: II,IU)')
    parser.add_argument('--channel', type=str, default='BHZ',
                       help='Channel code (default: BHZ)')
    parser.add_argument('--min-dist', type=float, default=30.0,
                       help='Minimum station distance in degrees (default: 30)')
    parser.add_argument('--max-dist', type=float, default=80.0,
                       help='Maximum station distance in degrees (default: 80)')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip download, only process existing data')
    
    # Other
    parser.add_argument('--demo', action='store_true',
                       help='Show demo and usage info')
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.start and args.end:
        # Automatic mode: fetch catalog and process
        full_automatic_pipeline(
            start_date=args.start,
            end_date=args.end,
            min_magnitude=args.min_mag,
            max_magnitude=args.max_mag,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            output_dir=args.output,
            networks=args.networks,
            channel=args.channel,
            min_distance=args.min_dist,
            max_distance=args.max_dist,
            source=args.source
        )
    elif args.catalog:
        # Use existing catalog
        process_catalog(
            catalog_file=args.catalog,
            output_dir=args.output,
            networks=args.networks,
            channel=args.channel,
            min_distance=args.min_dist,
            max_distance=args.max_dist,
            download=not args.no_download
        )
    else:
        parser.print_help()
        print("\n" + "="*70)
        print("QUICK START:")
        print("="*70)
        print("\n1. AUTOMATIC (recommended):")
        print("   python iris_downloader.py --start 2023-01-01 --end 2023-12-31 --min-mag 6.5")
        print("\n2. With your own catalog:")
        print("   python iris_downloader.py --catalog my_events.csv")
        print("\n3. See demo:")
        print("   python iris_downloader.py --demo")
        print("="*70)

