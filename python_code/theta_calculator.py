#!/usr/bin/env python3
"""
THETA Calculator - Seismic Energy and Ray Parameter Analysis

Python implementation of Fortran seismological routines for computing:
- P-wave ray parameter p
- Seismic radiated energy E
- Theta parameter Θ = log10(E/M0)

Based on the methodology of Boatwright and Choy (1986).

Usage:
    python theta_calculator.py --help
    python theta_calculator.py --demo
    python theta_calculator.py --data <file> --response <file> --epicenter <file>

Author: Converted from Fortran code
Date: 2025
"""

import argparse
import numpy as np
import sys
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


def print_banner():
    """Print program banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    THETA CALCULATOR                           ║
║         Seismic Energy & Ray Parameter Analysis               ║
╠═══════════════════════════════════════════════════════════════╣
║  Computes:                                                    ║
║    • P-wave ray parameter p                                   ║
║    • Radiated seismic energy E                                ║
║    • Energy-to-moment ratio Θ = log10(E/M0)                   ║
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="THETA Calculator - Seismic Energy and Ray Parameter Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python theta_calculator.py --demo
  python theta_calculator.py --data record.dat --response resp.dat --epicenter epi.dat

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
    
    args = parser.parse_args()
    
    if args.demo:
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
