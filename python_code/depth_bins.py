"""
Depth Bin Parameters - Based on Saloor and Okal (2018)

Different depth ranges require different correction parameters for:
- Distance correction coefficients (a0, a1, a2) for estimated FgP²
- t* attenuation factor (gamma_h)
- Time window duration

Depth Bins:
- SHALLOW: 0-80 km (original Boatwright & Choy)
- I_1: 80-135 km (Intermediate 1)
- I_2: 135-300 km (Intermediate 2)  
- D_1: 300-450 km (Deep 1)
- D_2: >450 km (Deep 2)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DepthBinParameters:
    """
    Parameters for a specific depth bin.
    
    Attributes
    ----------
    name : str
        Bin name (SHALLOW, I_1, I_2, D_1, D_2)
    depth_min : float
        Minimum depth for this bin (km)
    depth_max : float
        Maximum depth for this bin (km)
    a0, a1, a2 : float
        Distance correction coefficients for FgP² estimation
        FgP² = a0 + a1*Δ + a2*Δ²
    gamma_h : float
        t* attenuation scaling factor
    window_base : float
        Base time window duration (seconds)
    window_slope : float
        Additional window duration per km depth (seconds/km)
    reference_depth : float
        Reference depth for window calculation (km)
    """
    name: str
    depth_min: float
    depth_max: float
    a0: float
    a1: float
    a2: float
    gamma_h: float
    window_base: float
    window_slope: float = 0.0
    reference_depth: float = 0.0


# Define all depth bins based on Saloor and Okal (2018)
DEPTH_BINS = {
    'SHALLOW': DepthBinParameters(
        name='SHALLOW',
        depth_min=0.0,
        depth_max=80.0,
        # Original Boatwright & Choy (1986) coefficients
        a0=1.171,
        a1=-7.271e-3,
        a2=6.009e-5,
        gamma_h=1.0,  # No scaling
        window_base=70.0,
        window_slope=0.0,
        reference_depth=0.0
    ),
    
    'I_1': DepthBinParameters(
        name='I_1',
        depth_min=80.0,
        depth_max=135.0,
        # Saloor and Okal, Eq. (15)
        a0=0.8450,
        a1=3.701e-3,
        a2=-4.335e-5,
        gamma_h=0.8,  # Reduced attenuation
        window_base=70.0,
        window_slope=0.0,
        reference_depth=0.0
    ),
    
    'I_2': DepthBinParameters(
        name='I_2',
        depth_min=135.0,
        depth_max=300.0,
        # Saloor and Okal, Eq. (16)
        a0=0.6210,
        a1=5.102e-3,
        a2=-3.891e-5,
        gamma_h=0.7,
        window_base=80.0,
        window_slope=0.0,
        reference_depth=0.0
    ),
    
    'D_1': DepthBinParameters(
        name='D_1',
        depth_min=300.0,
        depth_max=450.0,
        # Saloor and Okal, Eq. (17)
        a0=0.2353,
        a1=4.109e-3,
        a2=-8.453e-6,
        gamma_h=0.6,
        # Window: 90 + 0.2*(depth - 300) seconds
        window_base=90.0,
        window_slope=0.2,
        reference_depth=300.0
    ),
    
    'D_2': DepthBinParameters(
        name='D_2',
        depth_min=450.0,
        depth_max=700.0,
        # Saloor and Okal, Eq. (18)
        a0=0.1850,
        a1=3.502e-3,
        a2=-5.210e-6,
        gamma_h=0.5,
        # Window: 120 + 0.25*(depth - 450) seconds
        window_base=120.0,
        window_slope=0.25,
        reference_depth=450.0
    ),
}


def get_depth_bin(depth_km: float) -> DepthBinParameters:
    """
    Get the appropriate depth bin parameters for a given depth.
    
    Parameters
    ----------
    depth_km : float
        Source depth in km
        
    Returns
    -------
    DepthBinParameters
        Parameters for the appropriate depth bin
        
    Raises
    ------
    ValueError
        If depth is outside valid range (0-700 km)
    """
    if depth_km < 0:
        raise ValueError(f"Depth cannot be negative: {depth_km} km")
    
    if depth_km > 700:
        raise ValueError(f"Depth too large for tables: {depth_km} km (max 700 km)")
    
    for bin_params in DEPTH_BINS.values():
        if bin_params.depth_min <= depth_km < bin_params.depth_max:
            return bin_params
    
    # Edge case: exactly 700 km
    return DEPTH_BINS['D_2']


def get_depth_bin_by_name(name: str) -> DepthBinParameters:
    """
    Get depth bin parameters by name.
    
    Parameters
    ----------
    name : str
        Bin name: 'SHALLOW', 'I_1', 'I_2', 'D_1', or 'D_2'
        
    Returns
    -------
    DepthBinParameters
    """
    name = name.upper()
    if name not in DEPTH_BINS:
        raise ValueError(f"Unknown depth bin: {name}. Valid bins: {list(DEPTH_BINS.keys())}")
    return DEPTH_BINS[name]


def estimated_FgP2_depth_corrected(distance_deg: float, depth_km: float) -> Tuple[float, str]:
    """
    Compute estimated (FgP)² with depth-dependent correction.
    
    Uses the appropriate coefficients for the depth bin.
    
    Parameters
    ----------
    distance_deg : float
        Epicentral distance in degrees
    depth_km : float
        Source depth in km
        
    Returns
    -------
    tuple
        (FgP2_estimated, bin_name)
    """
    bin_params = get_depth_bin(depth_km)
    
    fgp2 = (bin_params.a0 + 
            bin_params.a1 * distance_deg + 
            bin_params.a2 * distance_deg * distance_deg)
    
    return fgp2, bin_params.name


def compute_tstar_depth_corrected(freq: float, depth_km: float) -> float:
    """
    Compute t* attenuation with depth-dependent scaling.
    
    Based on Saloor and Okal (2018), Equation (14).
    t* is scaled by gamma_h factor that depends on depth.
    
    Parameters
    ----------
    freq : float
        Frequency in Hz
    depth_km : float
        Source depth in km
        
    Returns
    -------
    float
        t* value in seconds
    """
    bin_params = get_depth_bin(depth_km)
    gamma_h = bin_params.gamma_h
    
    # Base t* calculation (Choy and Cormier, 1986)
    if freq < 0.1:
        if bin_params.name == 'SHALLOW':
            tstar_base = 0.9 - 0.1 * np.log10(freq)
        else:
            # Modified for intermediate/deep (Saloor & Okal)
            tstar_base = gamma_h * (0.9 - 0.1 * np.log10(freq))
    elif freq < 1.0:
        if bin_params.name == 'SHALLOW':
            tstar_base = 0.5 - 0.5 * np.log10(freq)
        else:
            tstar_base = gamma_h * (0.4 - 0.6 * np.log10(freq))
    else:
        if bin_params.name == 'SHALLOW':
            tstar_base = 0.5 - 0.1 * np.log10(freq)
        else:
            tstar_base = gamma_h * (0.4 - 0.1 * np.log10(freq))
    
    return tstar_base


def compute_time_window(depth_km: float, moment: float = 1.0e27) -> float:
    """
    Compute appropriate time window duration for depth.
    
    Based on Saloor and Okal (2018), Equation (13).
    
    Parameters
    ----------
    depth_km : float
        Source depth in km
    moment : float
        Seismic moment (for large event adjustment)
        
    Returns
    -------
    float
        Window duration in seconds
    """
    bin_params = get_depth_bin(depth_km)
    
    # Base window
    window = bin_params.window_base
    
    # Depth-dependent adjustment
    if bin_params.window_slope > 0:
        window += bin_params.window_slope * (depth_km - bin_params.reference_depth)
    
    # Large event adjustment
    if moment > 1.0e29:
        window += 80.0  # Additional time for very large events
    
    return window


def print_depth_bin_info(depth_km: float):
    """
    Print information about the depth bin for a given depth.
    
    Parameters
    ----------
    depth_km : float
        Source depth in km
    """
    bin_params = get_depth_bin(depth_km)
    
    print(f"\n{'='*60}")
    print(f"DEPTH BIN INFORMATION")
    print(f"{'='*60}")
    print(f"  Source depth: {depth_km:.1f} km")
    print(f"  Depth bin: {bin_params.name}")
    print(f"  Valid range: {bin_params.depth_min:.0f} - {bin_params.depth_max:.0f} km")
    print(f"\n  Distance correction coefficients:")
    print(f"    a0 = {bin_params.a0:.4f}")
    print(f"    a1 = {bin_params.a1:.4e}")
    print(f"    a2 = {bin_params.a2:.4e}")
    print(f"\n  Attenuation factor (γh): {bin_params.gamma_h}")
    print(f"  Time window base: {bin_params.window_base:.0f} s")
    if bin_params.window_slope > 0:
        print(f"  Time window slope: +{bin_params.window_slope:.2f} s/km")
    print(f"{'='*60}\n")


def validate_depth_for_bin(depth_km: float, expected_bin: str) -> bool:
    """
    Validate that a depth falls within expected bin.
    
    Parameters
    ----------
    depth_km : float
        Source depth in km
    expected_bin : str
        Expected bin name
        
    Returns
    -------
    bool
        True if depth is in expected bin
    """
    try:
        actual_bin = get_depth_bin(depth_km)
        return actual_bin.name == expected_bin.upper()
    except ValueError:
        return False


# Summary table for reference
DEPTH_BIN_SUMMARY = """
╔═══════════════════════════════════════════════════════════════════════╗
║                    DEPTH BIN PARAMETERS                               ║
║                 (Saloor and Okal, 2018)                               ║
╠═══════════╦═══════════╦═══════════════════════════╦═══════╦══════════╣
║ Bin       ║ Depth(km) ║ FgP² = a0 + a1*Δ + a2*Δ² ║  γh   ║ Window   ║
╠═══════════╬═══════════╬═══════════════════════════╬═══════╬══════════╣
║ SHALLOW   ║   0-80    ║ 1.171 - 7.27e-3Δ + 6.0e-5Δ² ║ 1.0   ║  70 s    ║
║ I_1       ║  80-135   ║ 0.845 + 3.70e-3Δ - 4.3e-5Δ² ║ 0.8   ║  70 s    ║
║ I_2       ║ 135-300   ║ 0.621 + 5.10e-3Δ - 3.9e-5Δ² ║ 0.7   ║  80 s    ║
║ D_1       ║ 300-450   ║ 0.235 + 4.11e-3Δ - 8.5e-6Δ² ║ 0.6   ║ 90+0.2h  ║
║ D_2       ║ 450-700   ║ 0.185 + 3.50e-3Δ - 5.2e-6Δ² ║ 0.5   ║ 120+0.25h║
╚═══════════╩═══════════╩═══════════════════════════╩═══════╩══════════╝

Where:
  Δ = epicentral distance (degrees)
  γh = t* attenuation scaling factor
  h = depth relative to bin minimum
"""


if __name__ == "__main__":
    print(DEPTH_BIN_SUMMARY)
    
    # Test different depths
    test_depths = [15, 100, 200, 350, 500]
    
    for depth in test_depths:
        print_depth_bin_info(depth)
        
        # Test FgP² at 60 degrees
        fgp2, bin_name = estimated_FgP2_depth_corrected(60.0, depth)
        print(f"  FgP² at 60°: {fgp2:.4f}")
        
        # Test t* at 0.1 Hz
        tstar = compute_tstar_depth_corrected(0.1, depth)
        print(f"  t* at 0.1 Hz: {tstar:.3f} s")
        
        # Test window
        window = compute_time_window(depth)
        print(f"  Time window: {window:.0f} s")
        print()

