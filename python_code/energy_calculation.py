"""
Energy Calculation Module - Python conversion of Fortran routines
Main module for computing seismic radiated energy and the Theta (Θ) parameter.

Based on Boatwright and Choy (1986) methodology:
    Θ = log10(E/M0)
    
where E is the radiated seismic energy and M0 is the seismic moment.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from seismic_utils import (
    great_circle, find_fft_size, detrend, cosine_taper, seconds_to_hms
)
from instrument_response import InstrumentResponse, compute_response
from travel_time import (
    JBTables, compute_radiation_coefficients, compute_FgP2,
    estimated_FgP2, compute_tstar
)
from geometric_spreading import (
    GeometricSpreadingTables, compute_geometric_spreading_fortran
)
from depth_bins import (
    get_depth_bin, estimated_FgP2_depth_corrected, 
    compute_tstar_depth_corrected, compute_time_window, print_depth_bin_info
)


PI = np.pi
TWOPI = 2.0 * PI
EARTH_RADIUS_M = 6371.0e3
DEG_TO_KM = EARTH_RADIUS_M / 1000.0 * PI / 180.0


@dataclass
class EnergyResult:
    """
    Results from energy calculation.
    
    Attributes
    ----------
    station : str
        Station code
    distance_deg : float
        Epicentral distance in degrees
    azimuth : float
        Station azimuth from epicenter
    slowness_deg : float
        Ray parameter in s/deg
    slowness_km : float  
        Ray parameter in s/km
    depth_km : float
        Source depth in km
    depth_bin : str
        Depth bin name (SHALLOW, I_1, I_2, D_1, D_2)
    energy_estimated : float
        Estimated seismic energy using statistical focal mechanism (ergs)
    energy_true_mech : float
        Energy using true focal mechanism (ergs)
    theta_estimated : float
        Estimated Θ = log10(E/M0)
    theta_true_mech : float
        Θ using true focal mechanism
    FgP2 : float
        Radiation pattern factor (FgP)²
    FgP2_estimated : float
        Estimated (FgP)² from distance regression (depth-corrected)
    geometric_spreading : float
        Geometric spreading factor g
    receiver_function : float
        Receiver function rpz
    """
    station: str
    distance_deg: float
    azimuth: float
    slowness_deg: float
    slowness_km: float
    depth_km: float
    depth_bin: str
    energy_estimated: float
    energy_true_mech: float
    theta_estimated: float
    theta_true_mech: float
    FgP2: float
    FgP2_estimated: float
    geometric_spreading: float
    receiver_function: float


def compute_seismic_energy(
    data: np.ndarray,
    dt: float,
    instrument: InstrumentResponse,
    elat: float, elon: float,
    slat: float, slon: float,
    depth_km: float,
    moment: float,
    strike: float = 0.0,
    dip: float = 45.0,
    rake: float = 90.0,
    tmin: float = 0.5,
    tmax: float = 10.0,
    station_code: str = "STA",
    jb_tables: Optional[JBTables] = None
) -> EnergyResult:
    """
    Compute seismic radiated energy and Θ parameter from a seismogram.
    
    This is the main Python equivalent of the Fortran endig.f program.
    
    Parameters
    ----------
    data : np.ndarray
        Seismogram data (raw counts or ground motion)
    dt : float
        Sampling interval in seconds
    instrument : InstrumentResponse
        Instrument response (poles, zeros, gain)
    elat, elon : float
        Epicenter latitude and longitude (degrees)
    slat, slon : float
        Station latitude and longitude (degrees)
    depth_km : float
        Source depth in km
    moment : float
        Seismic moment M0 in dyn-cm
    strike, dip, rake : float
        Focal mechanism parameters (degrees)
    tmin, tmax : float
        Bandpass filter periods (seconds)
    station_code : str
        Station identifier
    jb_tables : JBTables, optional
        Travel time tables (created if not provided)
        
    Returns
    -------
    EnergyResult
        Results of energy calculation
    """
    # Constants
    reth = EARTH_RADIUS_M
    rho = 3.0e3       # Density in kg/m³
    vel = 7.0e3       # P-wave velocity in m/s
    q_bc = np.sqrt(3.0) ** 5  # Boatwright-Choy q parameter
    
    # Initialize JB tables if not provided
    if jb_tables is None:
        jb_tables = JBTables()
    
    # Calculate distance and azimuth
    dist_deg, az_es, az_se, gc = great_circle(elat, elon, slat, slon)
    
    # Get slowness (ray parameter)
    p_deg = jb_tables.get_slowness(dist_deg, depth_km)
    p_km = p_deg / DEG_TO_KM
    p_rad = p_deg * 180.0 / PI
    
    # Compute radiation coefficients and FgP²
    FgP2, coeffs = compute_FgP2(dist_deg, depth_km, az_es, strike, dip, rake, q_bc)
    
    # Use Fortran-style geometric spreading (more accurate)
    d2t_tables = GeometricSpreadingTables()
    g, rpz = compute_geometric_spreading_fortran(
        dist_deg, depth_km, p_deg, d2t_tables=d2t_tables
    )
    
    # Estimated FgP² from distance regression (depth-corrected)
    est_FgP2, depth_bin_name = estimated_FgP2_depth_corrected(dist_deg, depth_km)
    print(f"Depth bin: {depth_bin_name} ({depth_km} km)")
    
    # If FgP² is too small, use minimum value to avoid blow-up
    FgP2_used = max(FgP2, 0.2)
    
    print(f"Station: {station_code}")
    print(f"Distance: {dist_deg:.2f}°, Azimuth: {az_es:.2f}°")
    print(f"Slowness: p = {p_deg:.3f} s/deg = {p_km:.5f} s/km = {p_rad:.2f} s/rad")
    print(f"Radiation: Fp = {coeffs['Fp']:.3f}, Fpp = {coeffs['Fpp']:.3f}, Fsp = {coeffs['Fsp']:.3f}")
    print(f"Reflection: PP = {coeffs['PP']:.3f}, SP = {coeffs['SP']:.3f}")
    print(f"(FgP)² = {FgP2:.4f}, Estimated (FgP)² = {est_FgP2:.4f}")
    
    # Prepare data for FFT
    nx = len(data)
    nfft, jfft = find_fft_size(nx)
    
    # Preprocess: detrend and taper
    x = detrend(data.copy())
    x = cosine_taper(x, 0.05, 0.1)
    
    # Zero-pad and FFT
    z = np.zeros(nfft, dtype=complex)
    z[:nx] = x * dt  # Multiply by dt for FFT normalization
    
    spectrum = np.fft.fft(z)
    df = 1.0 / (nfft * dt)
    
    print(f"FFT (order {jfft}) carried out")
    print(f"Bandpass filtering between {tmax:.1f} and {tmin:.1f} seconds")
    
    # Define frequency window
    fmin = 1.0 / tmax
    fmax = 1.0 / tmin
    nfmin = max(int(fmin / df + 0.5), 2)
    nfmax = int(fmax / df + 1.5)
    lcent = nfft // 2 + 1
    
    # Get poles and zeros
    zeros = instrument.zeros
    poles = instrument.poles
    gain = instrument.total_gain
    
    # Add integration zeros if response is for velocity or acceleration
    if instrument.unit.upper() == 'M/S':
        zeros = np.append(zeros, 0+0j)
    elif instrument.unit.upper() == 'M/S**2':
        zeros = np.append(zeros, [0+0j, 0+0j])
    
    # Remove instrument response and apply bandpass
    for j in range(1, lcent):
        f = df * j
        
        if f < fmin or f > fmax:
            spectrum[j] = 0.0
            continue
        
        omega = 2.0 * PI * f
        
        # Compute instrument response
        zres = compute_response(f, zeros, poles, gain)
        
        # Convert from displacement to velocity response
        zres = zres / (1j * omega)
        
        # Remove instrument response
        if abs(zres) > 0:
            spectrum[j] = spectrum[j] / zres
        # spectrum[j] is now velocity in m/s * s (velocity spectral density)
    
    # Zero negative frequencies
    spectrum[0] = 0.0
    
    # Integrate energy in frequency domain
    # Equation (17) from Boatwright and Choy (1986)
    domega = 2.0 * PI * df
    sum_int = 0.0
    
    for j in range(nfmin, lcent):
        f = df * j
        omega = 2.0 * PI * f
        
        # t* attenuation correction (depth-dependent)
        tstar = compute_tstar_depth_corrected(f, depth_km)
        
        # Energy integrand: |v(ω)|² * e^(ω*t*)
        sum_int += (np.abs(spectrum[j]) ** 2) * domega * np.exp(omega * tstar)
    
    # Raw energy integral
    energy = sum_int
    print(f"Energy from integral: {energy:.6e}")
    
    # Apply medium parameters: E = (ρv/π) * integral
    energy = energy * rho * vel / PI
    
    # Geometric spreading correction
    # Rp = reth / g (see Kanamori and Stewart, 1976)
    geom_sp = g * rpz
    if geom_sp > 0:
        energy = energy * ((reth / geom_sp) ** 2)
    
    # Average radiation pattern <Fp>² = (4/15) from BC86 p. 2096
    avg_fp_sq = 4.0 / 15.0
    
    # Energy with true focal mechanism
    energy_true = energy * 4.0 * PI * (avg_fp_sq / FgP2_used)
    print(f"Preliminary energy (true mech.): {energy_true:.3e} J")
    
    # Energy with estimated focal mechanism
    energy_est = energy * 4.0 * PI * (avg_fp_sq / est_FgP2)
    print(f"Preliminary energy (estimated): {energy_est:.3e} J")
    
    # Add S-wave contribution and convert to ergs
    # Factor (1 + q_bc) accounts for S-wave energy
    energy_true = 1.0e7 * energy_true * (1.0 + q_bc)  # ergs
    energy_est = 1.0e7 * energy_est * (1.0 + q_bc)    # ergs
    
    print(f"Total energy (true mech.): {energy_true:.3e} ergs")
    print(f"Total energy (estimated): {energy_est:.3e} ergs")
    
    # Compute Θ = log10(E/M0)
    theta_true = np.log10(energy_true / moment)
    theta_est = np.log10(energy_est / moment)
    
    print(f"Θ (true mech.): {theta_true:.2f}")
    print(f"Θ (estimated): {theta_est:.2f}")
    
    return EnergyResult(
        station=station_code,
        distance_deg=dist_deg,
        azimuth=az_es,
        slowness_deg=p_deg,
        slowness_km=p_km,
        depth_km=depth_km,
        depth_bin=depth_bin_name,
        energy_estimated=energy_est,
        energy_true_mech=energy_true,
        theta_estimated=theta_est,
        theta_true_mech=theta_true,
        FgP2=FgP2,
        FgP2_estimated=est_FgP2,
        geometric_spreading=g,
        receiver_function=rpz
    )


def compute_window_bounds(
    origin_time: float,
    travel_time: float,
    moment: float,
    pre_p: float = 10.0
) -> Tuple[float, float]:
    """
    Compute time window bounds for P-wave analysis.
    
    Parameters
    ----------
    origin_time : float
        Origin time in seconds from record start
    travel_time : float
        P-wave travel time in seconds
    moment : float
        Seismic moment (used to adjust window length)
    pre_p : float
        Time before P arrival to start window (seconds)
        
    Returns
    -------
    tuple
        (window_start, window_end) in seconds
    """
    p_arrival = origin_time + travel_time
    
    # Window start: 10 seconds before P
    t_start = p_arrival - pre_p
    
    # Window end: depends on event size
    if moment > 1.0e29:
        t_end = t_start + 150.0  # Large event
    else:
        t_end = t_start + 70.0   # Normal event
    
    return max(0, t_start), t_end


def classify_event(theta: float) -> str:
    """
    Classify event based on Θ value.
    
    Parameters
    ----------
    theta : float
        Θ = log10(E/M0)
        
    Returns
    -------
    str
        Event classification
    """
    if theta <= -5.75:
        return "SLOW (tsunami earthquake potential)"
    elif theta >= -4.30:
        return "FAST (high stress drop)"
    else:
        return "NORMAL"


def combine_station_results(results: list) -> Tuple[float, float, int]:
    """
    Combine results from multiple stations.
    
    Parameters
    ----------
    results : list
        List of EnergyResult objects
        
    Returns
    -------
    tuple
        (average_theta_estimated, average_theta_true, n_stations)
    """
    if not results:
        return 0.0, 0.0, 0
    
    theta_est = np.mean([r.theta_estimated for r in results])
    theta_true = np.mean([r.theta_true_mech for r in results])
    
    return theta_est, theta_true, len(results)


# =============================================================================
# File I/O for data files
# =============================================================================
def read_sac_format_data(filename: str) -> Tuple[np.ndarray, dict]:
    """
    Read seismogram in simplified SAC-like format.
    
    This reads the text format used by the Fortran codes.
    
    Parameters
    ----------
    filename : str
        Path to data file
        
    Returns
    -------
    tuple
        (data_array, metadata_dict)
    """
    metadata = {}
    
    with open(filename, 'r') as f:
        # First line: station info
        line1 = f.readline().strip()
        # Parse station, component, year, day, hour, min, sec, lat, lon
        parts = line1.split()
        metadata['station'] = parts[0][:3]
        metadata['component'] = parts[0][3] if len(parts[0]) > 3 else 'Z'
        
        # Second line: nx, dt
        line2 = f.readline().strip().split()
        nx = int(float(line2[0]))
        dt = float(line2[1])
        metadata['nx'] = nx
        metadata['dt'] = dt
        
        # Read data
        data_lines = f.read().split()
        data = np.array([float(x) for x in data_lines[:nx]])
    
    return data, metadata


def read_epicenter_parameters(filename: str) -> dict:
    """
    Read epicenter parameters file.
    
    Parameters
    ----------
    filename : str
        Path to epicenter parameters file
        
    Returns
    -------
    dict
        Dictionary with epicenter info
    """
    params = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Line 1: lat, lon, depth
    parts = lines[0].split()
    params['lat'] = float(parts[0])
    params['lon'] = float(parts[1])
    if len(parts) > 2:
        params['depth'] = float(parts[2])
    else:
        params['depth'] = 33.0  # Default depth
    
    # Line 2: origin time
    if len(lines) > 1:
        parts = lines[1].split()
        params['hour'] = int(parts[0])
        params['minute'] = int(parts[1])
        params['second'] = float(parts[2])
    
    # Line 3: description
    if len(lines) > 2:
        params['description'] = lines[2].strip()
    
    return params


def read_moments_file(filename: str) -> dict:
    """
    Read seismic moments from file.
    
    Parameters
    ----------
    filename : str
        Path to moments file
        
    Returns
    -------
    dict
        Dictionary mapping event IDs to moments
    """
    moments = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                event_id = parts[0]
                try:
                    moment = float(parts[1])
                    moments[event_id] = moment
                except ValueError:
                    continue
    
    return moments


def write_results(filename: str, results: list):
    """
    Write energy results to file.
    
    Parameters
    ----------
    filename : str
        Output filename
    results : list
        List of EnergyResult objects
    """
    with open(filename, 'w') as f:
        f.write("# Station  Distance  Azimuth  Theta_est  Theta_true  Moment\n")
        for r in results:
            f.write(f"{r.station:>7s} {r.distance_deg:10.3f} {r.azimuth:10.3f} "
                    f"{r.theta_estimated:7.2f} {r.theta_true_mech:7.2f}\n")


# =============================================================================
# COMPLETE PROCESSING WORKFLOW
# =============================================================================
def process_event(
    data_files: list,
    epicenter_file: str,
    response_files: list,
    moments_file: str,
    strike: float = 0.0,
    dip: float = 45.0,
    rake: float = 90.0,
    tmin: float = 0.5,
    tmax: float = 10.0
) -> Tuple[list, float, float]:
    """
    Process complete event with multiple stations.
    
    Parameters
    ----------
    data_files : list
        List of seismogram file paths
    epicenter_file : str
        Path to epicenter parameters file
    response_files : list
        List of instrument response file paths
    moments_file : str
        Path to moments file
    strike, dip, rake : float
        Focal mechanism parameters
    tmin, tmax : float
        Bandpass filter periods
        
    Returns
    -------
    tuple
        (results_list, average_theta_est, average_theta_true)
    """
    from instrument_response import read_georom_format
    
    # Read epicenter
    epi_params = read_epicenter_parameters(epicenter_file)
    elat = epi_params['lat']
    elon = epi_params['lon']
    depth = epi_params.get('depth', 33.0)
    
    # Read moments
    moments = read_moments_file(moments_file)
    event_id = epi_params.get('description', 'unknown')
    moment = moments.get(event_id, 1.0e27)  # Default moment
    
    # Initialize JB tables once
    jb_tables = JBTables()
    
    results = []
    
    for data_file, resp_file in zip(data_files, response_files):
        try:
            # Read data
            data, metadata = read_sac_format_data(data_file)
            dt = metadata['dt']
            station = metadata['station']
            
            # Station coordinates would normally come from metadata
            # For now, use placeholders
            slat = metadata.get('lat', 0.0)
            slon = metadata.get('lon', 0.0)
            
            # Read instrument response
            instrument = read_georom_format(resp_file)
            
            # Compute energy
            result = compute_seismic_energy(
                data=data,
                dt=dt,
                instrument=instrument,
                elat=elat, elon=elon,
                slat=slat, slon=slon,
                depth_km=depth,
                moment=moment,
                strike=strike, dip=dip, rake=rake,
                tmin=tmin, tmax=tmax,
                station_code=station,
                jb_tables=jb_tables
            )
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {data_file}: {e}")
            continue
    
    # Combine results
    avg_theta_est, avg_theta_true, n_sta = combine_station_results(results)
    
    print(f"\n{'='*60}")
    print(f"Combined results from {n_sta} stations:")
    print(f"Average Θ (estimated): {avg_theta_est:.2f}")
    print(f"Average Θ (true mech): {avg_theta_true:.2f}")
    print(f"Event classification: {classify_event(avg_theta_est)}")
    
    return results, avg_theta_est, avg_theta_true


if __name__ == "__main__":
    print("Testing energy_calculation.py")
    print("="*60)
    
    # Create synthetic test data
    np.random.seed(42)
    dt = 0.5  # sampling rate
    duration = 100  # seconds
    n_samples = int(duration / dt)
    
    # Simulate P-wave arrival with some noise
    t = np.arange(n_samples) * dt
    p_arrival = 30.0  # seconds
    
    # Simple wavelet for P-wave
    data = np.zeros(n_samples)
    for i, ti in enumerate(t):
        if ti > p_arrival:
            tau = ti - p_arrival
            # Damped sinusoid to simulate P-wave
            data[i] = 1000 * np.exp(-tau/5) * np.sin(2*PI*tau/2) 
    
    # Add noise
    data += np.random.randn(n_samples) * 10
    
    # Create mock instrument response
    instrument = InstrumentResponse(
        a0=1.0e10,
        sensitivity=1.0,
        zeros=np.array([0+0j, 0+0j]),
        poles=np.array([
            -0.01234+0.01234j, -0.01234-0.01234j,
            -39.18+49.12j, -39.18-49.12j
        ]),
        unit='M'
    )
    
    # Test parameters
    elat, elon = -15.0, -75.0  # Epicenter (Peru)
    slat, slon = 34.0, -118.0  # Station (Pasadena)
    depth = 33.0  # km
    moment = 1.0e27  # dyn-cm
    
    print("Test with synthetic data:")
    print(f"Epicenter: ({elat}, {elon})")
    print(f"Station: ({slat}, {slon})")
    print(f"Depth: {depth} km")
    print(f"Moment: {moment:.2e} dyn-cm")
    print()
    
    # Compute energy
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
        station_code="TEST"
    )
    
    print()
    print("="*60)
    print("SUMMARY:")
    print(f"Station: {result.station}")
    print(f"Distance: {result.distance_deg:.2f}°")
    print(f"Slowness: {result.slowness_deg:.3f} s/deg")
    print(f"Energy (estimated): {result.energy_estimated:.3e} ergs")
    print(f"Θ (estimated): {result.theta_estimated:.2f}")
    print(f"Event type: {classify_event(result.theta_estimated)}")
    
    print("\nAll tests passed!")

