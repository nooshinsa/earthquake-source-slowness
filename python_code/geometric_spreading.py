#!/usr/bin/env python3
"""
Geometric Spreading Module

Reads d²T/dΔ² tables from Fortran files and computes
accurate geometric spreading and receiver functions.

Based on Fortran subroutine GEORPZ in endig.f
"""

import numpy as np
import os

# Earth parameters
EARTH_RADIUS_KM = 6371.0


class GeometricSpreadingTables:
    """
    Load and interpolate d²T/dΔ² tables for geometric spreading calculation.
    """
    
    def __init__(self, d2t_file: str = None):
        """
        Initialize with path to d2tdd2.p file.
        
        Parameters
        ----------
        d2t_file : str
            Path to d2tdd2.p file (from Fortran)
        """
        # Default path - look in parent directory
        if d2t_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(current_dir, 'd2tdd2.p'),
                os.path.join(current_dir, '..', 'd2tdd2.p'),
                os.path.join(current_dir, '..', 'THETHA', 'd2tdd2.p'),
            ]
            d2t_file = next((path for path in candidates if os.path.exists(path)),
                            candidates[-1])
        
        self.d2t_file = d2t_file
        self.depths = None
        self.distances = None
        self.d2t_table = None
        self._table_loaded = False
        
        if os.path.exists(d2t_file):
            self._load_table()
        else:
            print(f"WARNING: d2tdd2.p not found at {d2t_file}")
            print("Using approximate geometric spreading values")
    
    def _load_table(self):
        """
        Load d²T/dΔ² table from Fortran file.
        """
        print(f"Loading d²T/dΔ² table from {self.d2t_file}")
        
        # The Fortran format is:
        # Line 1: Header (skip)
        # Line 2: Depths - 5x (skip 5 chars), then 15f5.1
        # Lines 3+: d²T/dΔ² values, 15 depth sections, each with 113 values
        
        with open(self.d2t_file, 'r') as f:
            lines = f.readlines()
        
        # Parse depths from line 2 (skip first 5 chars, then 15 values of width 5)
        depth_line = lines[1]
        # Skip first 5 characters, then read 15 values each 5 chars wide
        depth_str = depth_line[5:]  # Skip 5x
        self.depths = []
        for i in range(15):
            try:
                val = float(depth_str[i*5:(i+1)*5])
                self.depths.append(val)
            except:
                pass
        self.depths = np.array(self.depths) if self.depths else np.array([0, 33, 100, 200, 300, 400, 500, 600, 700])
        
        # Standard JB distances (113 values)
        self.distances = np.concatenate([
            np.arange(0, 10.5, 0.5),     # 0, 0.5, 1, ..., 10 (21 values)
            np.arange(11, 101),           # 11, 12, ..., 100 (90 values)  
            np.arange(102, 114, 2)        # 102, 104, 106, 108, 110, 112 (6 values, but may be fewer)
        ])
        # Ensure exactly 113 distances
        if len(self.distances) > 113:
            self.distances = self.distances[:113]
        
        # Parse all d²T/dΔ² values from remaining lines
        all_values = []
        for line in lines[2:]:  # Skip header and depth line
            # Parse Fortran scientific notation
            # Format is typically 5 values per line in E format
            line = line.strip()
            if not line:
                continue
            # Replace Fortran D notation with E
            line = line.replace('D', 'E').replace('d', 'e')
            parts = line.split()
            for part in parts:
                try:
                    all_values.append(float(part))
                except:
                    pass
        
        # Organize into table: 15 depths x 113 distances
        # File is organized as 15 sections (one per depth), each with 113 values
        n_depths = len(self.depths)
        n_distances = 113
        
        self.d2t_table = np.zeros((n_depths, n_distances))
        
        values_per_depth = n_distances
        for idep in range(min(n_depths, len(all_values) // values_per_depth)):
            start = idep * values_per_depth
            end = start + values_per_depth
            if end <= len(all_values):
                self.d2t_table[idep, :] = all_values[start:end]
        
        # Mark table as loaded
        self._table_loaded = True
        
        print(f"Loaded: {n_depths} depths x {n_distances} distances")
        print(f"Depths: {self.depths[:5]}... (km)")
        print(f"Sample d²T/dΔ² at 60°, shallow: {self.d2t_table[0, 70] if len(self.d2t_table) > 0 else 'N/A'}")
    
    def get_d2t_dd2(self, distance_deg: float, depth_km: float) -> float:
        """
        Get d²T/dΔ² by interpolation.
        
        Parameters
        ----------
        distance_deg : float
            Epicentral distance in degrees
        depth_km : float
            Source depth in km
            
        Returns
        -------
        float
            d²T/dΔ² in s/deg²
        """
        if not self._table_loaded or self.d2t_table is None:
            return self._approximate_d2t_dd2(distance_deg, depth_km)
        
        # Bilinear interpolation
        # Find bracketing indices
        idep = np.searchsorted(self.depths, depth_km) - 1
        idep = max(0, min(idep, len(self.depths) - 2))
        
        idis = np.searchsorted(self.distances, distance_deg) - 1
        idis = max(0, min(idis, len(self.distances) - 2))
        
        # Interpolation weights
        h1 = self.depths[idep]
        h2 = self.depths[idep + 1]
        d1 = self.distances[idis]
        d2 = self.distances[idis + 1]
        
        wh = (depth_km - h1) / (h2 - h1) if h2 != h1 else 0
        wd = (distance_deg - d1) / (d2 - d1) if d2 != d1 else 0
        
        # Bilinear interpolation
        v00 = self.d2t_table[idep, idis]
        v01 = self.d2t_table[idep, idis + 1]
        v10 = self.d2t_table[idep + 1, idis]
        v11 = self.d2t_table[idep + 1, idis + 1]
        
        value = (v00 * (1 - wh) * (1 - wd) +
                 v01 * (1 - wh) * wd +
                 v10 * wh * (1 - wd) +
                 v11 * wh * wd)
        
        return value
    
    def _approximate_d2t_dd2(self, distance_deg: float, depth_km: float) -> float:
        """
        Approximate d²T/dΔ² when tables not available.
        
        Based on inspection of d2tdd2.p values.
        """
        # Values from d2tdd2.p file inspection
        # For mantle P waves at 30-80°, typical values are around -0.07 s/deg²
        # (converted to s/rad²)
        
        if distance_deg < 20:
            return -0.02
        elif distance_deg < 30:
            return -0.04
        elif distance_deg < 40:
            return -0.05
        elif distance_deg < 60:
            return -0.06
        elif distance_deg < 80:
            return -0.07
        else:
            return -0.075


def compute_geometric_spreading_fortran(
    distance_deg: float,
    depth_km: float,
    slowness_deg: float,
    alpha_receiver: float = 7.0,
    beta_receiver: float = 4.04,
    rho_receiver: float = 3.0,
    alpha_source: float = 6.5,
    rho_source: float = 2.9,
    d2t_tables: GeometricSpreadingTables = None
) -> tuple:
    """
    Compute geometric spreading and receiver function.
    
    Exact translation of Fortran GEORPZ subroutine.
    
    Parameters
    ----------
    distance_deg : float
        Epicentral distance in degrees
    depth_km : float
        Source depth in km
    slowness_deg : float
        Ray parameter in s/deg
    alpha_receiver, beta_receiver, rho_receiver : float
        Receiver parameters
    alpha_source, rho_source : float
        Source region parameters
    d2t_tables : GeometricSpreadingTables, optional
        Pre-loaded d²T/dΔ² tables
        
    Returns
    -------
    tuple
        (g, rpz) - geometric spreading factor and receiver function
    """
    reth = EARTH_RADIUS_KM
    pi = np.pi
    
    # Convert slowness to s/rad then to s/km
    p_rad = slowness_deg * 180.0 / pi  # s/rad
    p = p_rad / reth  # s/km
    
    # Earth flattening correction
    error = reth / (reth - depth_km)
    
    # Incidence angle at source
    sin_ih = p * alpha_source
    if abs(sin_ih) > 1.0:
        sin_ih = np.sign(sin_ih)
    angih = np.arcsin(sin_ih)
    cosih = np.cos(angih)
    
    # Get d²T/dΔ² from tables or approximation
    if d2t_tables is not None:
        d2tdd2 = d2t_tables.get_d2t_dd2(distance_deg, depth_km)
    else:
        # Use approximation
        d2tdd2 = _default_d2t_dd2(distance_deg, depth_km)
    
    # Convert d²T/dΔ² from s/deg² to s/rad²
    d2tdd2 = d2tdd2 * (180.0 / pi) ** 2
    
    # d(ih)/d(delta)
    dihdel = (d2tdd2 / cosih) * alpha_source / (reth - depth_km)
    dihdel = abs(dihdel)
    
    # Incidence angle at receiver (with flattening correction)
    sin_i0 = p * alpha_receiver / error
    if abs(sin_i0) > 1.0:
        sin_i0 = np.sign(sin_i0)
    angi0 = np.arcsin(sin_i0)
    cos_i0 = np.cos(angi0)
    
    # Geometric spreading factor (Kanamori & Stewart, 1976)
    xonst = pi / 180.0  # degrees to radians
    factor = p * rho_source * alpha_source**2 / (rho_receiver * alpha_receiver)
    
    sin_delta = np.sin(distance_deg * xonst)
    if sin_delta < 1e-10:
        sin_delta = 1e-10
    
    g = np.sqrt(factor * dihdel / (sin_delta * cos_i0))
    
    # Receiver function (Helmberger, 1974)
    p2 = p**2 / error**2
    betar2 = beta_receiver**2
    
    etaa = np.sqrt(max(0, 1.0/(alpha_receiver**2) - p2))
    etab = np.sqrt(max(0, 1.0/betar2 - p2))
    
    denom = (betar2 * (etab**2 - p2)**2 + 4.0 * p2 * betar2 * etaa * etab)
    if abs(denom) < 1e-20:
        denom = 1e-20
    
    rpz = 2.0 * etaa * (etab**2 - p2) / denom
    rpz = alpha_receiver * rpz
    
    return g, rpz


def _default_d2t_dd2(distance_deg: float, depth_km: float) -> float:
    """
    Default approximation for d²T/dΔ² when tables not loaded.
    
    Values extracted from d2tdd2.p file for shallow depths (0-33 km).
    These are in s/deg² units.
    """
    # Values from d2tdd2.p file for shallow depths, at key distances
    # Format: distance -> d²T/dΔ²
    # These are approximately correct for the 30-80° teleseismic range
    
    d2t_table = {
        30: -0.036,
        35: -0.041,
        40: -0.053,
        45: -0.056,
        50: -0.060,
        55: -0.063,
        60: -0.067,
        65: -0.070,
        70: -0.072,
        75: -0.073,
        80: -0.073,
        85: -0.074,
        90: -0.077
    }
    
    # Linear interpolation
    distances = sorted(d2t_table.keys())
    
    if distance_deg <= distances[0]:
        return d2t_table[distances[0]]
    if distance_deg >= distances[-1]:
        return d2t_table[distances[-1]]
    
    # Find bracketing distances
    for i in range(len(distances) - 1):
        d1, d2 = distances[i], distances[i+1]
        if d1 <= distance_deg <= d2:
            # Linear interpolation
            frac = (distance_deg - d1) / (d2 - d1)
            return d2t_table[d1] + frac * (d2t_table[d2] - d2t_table[d1])
    
    return -0.07  # Default fallback


if __name__ == "__main__":
    # Test
    print("Testing geometric spreading calculation")
    print("="*60)
    
    # Santa Cruz event parameters
    distance = 66.78  # degrees
    depth = 20.2      # km
    slowness = 4.693  # s/deg
    
    print(f"Distance: {distance}°")
    print(f"Depth: {depth} km")
    print(f"Slowness: {slowness} s/deg")
    
    # Try to load tables
    tables = GeometricSpreadingTables()
    
    # Compute
    g, rpz = compute_geometric_spreading_fortran(
        distance, depth, slowness, d2t_tables=tables
    )
    
    print(f"\nResults:")
    print(f"  Geometric spreading g = {g:.4f}")
    print(f"  Receiver function rpz = {rpz:.4f}")
    print(f"  geomsp = g * rpz = {g * rpz:.4f}")
    
    # Compare with what Fortran should give
    print("\n" + "="*60)
    print("RUN YOUR FORTRAN CODE AND COMPARE:")
    print("="*60)
    print("Look for these lines in Fortran output:")
    print("  'Surface response : X.XXX'")
    print("  'Geometrical spreading : X.XXX'")
    print("="*60)
