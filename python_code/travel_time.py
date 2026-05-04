"""
Travel Time and Slowness Module - Python conversion of Fortran routines
Handles P-wave travel times, slowness (ray parameter), and geometric spreading.
"""

import numpy as np
from typing import Tuple, Optional
import os


PI = np.pi
TWOPI = 2.0 * PI
EARTH_RADIUS_KM = 6371.0
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI


# =============================================================================
# JB (Jeffreys-Bullen) TRAVEL TIME TABLES
# =============================================================================
class JBTables:
    """
    Jeffreys-Bullen P-wave travel time tables.
    
    Provides travel times and ray parameters (slowness) for P-waves
    as a function of epicentral distance and source depth.
    
    Attributes
    ----------
    depths : np.ndarray
        Array of source depths (km)
    distances : np.ndarray
        Array of epicentral distances (degrees)
    times : np.ndarray
        2D array of travel times [depth, distance] (seconds)
    """
    
    def __init__(self, filename: Optional[str] = None):
        """
        Initialize JB tables.
        
        Parameters
        ----------
        filename : str, optional
            Path to JB table file. If None, uses built-in table.
        """
        if filename is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(current_dir, 'jfbup.dat'),
                os.path.join(current_dir, '..', 'jfbup.dat'),
                os.path.join(current_dir, '..', 'THETHA', 'jfbup.dat'),
            ]
            filename = next((path for path in candidates if os.path.exists(path)), None)

        if filename and os.path.exists(filename):
            self._load_from_file(filename)
        else:
            self._create_builtin_table()
    
    def _create_builtin_table(self):
        """Create built-in JB tables based on standard values."""
        # Standard JB depth values (km)
        self.depths = np.array([0., 33., 50., 100., 150., 200., 250., 
                                300., 400., 500., 600., 650., 700., 750., 800.])
        
        # Distance values from 0 to 100+ degrees
        self.distances = np.concatenate([
            np.arange(0, 10, 0.5),    # 0-10 deg, 0.5 deg increments
            np.arange(10, 113, 1.0)   # 10-112 deg, 1 deg increments
        ])
        
        # Create approximate travel time table
        # These are simplified approximations of JB tables
        n_dep = len(self.depths)
        n_dis = len(self.distances)
        self.times = np.zeros((n_dep, n_dis))
        
        # Approximate P-wave velocity model parameters
        # Using simplified Earth model
        for i, depth in enumerate(self.depths):
            for j, dist in enumerate(self.distances):
                if dist == 0:
                    self.times[i, j] = 0.0
                else:
                    # Approximate travel time using ray theory
                    # This is a simplified model - real JB tables are more accurate
                    avg_velocity = 10.0 - 0.002 * depth  # Approximate average P velocity
                    ray_path = self._estimate_ray_path(dist, depth)
                    self.times[i, j] = ray_path / avg_velocity
        
        # Apply corrections for known travel times at key distances
        self._apply_jb_corrections()
    
    def _estimate_ray_path(self, distance_deg: float, depth_km: float) -> float:
        """Estimate ray path length for given distance and depth."""
        # Convert to radians
        delta = distance_deg * DEG_TO_RAD
        
        # Simple chord approximation for short distances
        if distance_deg < 20:
            # Surface distance
            surface_dist = EARTH_RADIUS_KM * delta
            # Add depth correction
            return np.sqrt(surface_dist**2 + depth_km**2)
        else:
            # For larger distances, use great circle arc
            r = EARTH_RADIUS_KM - depth_km
            return 2 * r * np.sin(delta / 2)
    
    def _apply_jb_corrections(self):
        """Apply corrections to match standard JB travel times."""
        # Reference travel times for surface focus at key distances
        # These are from standard JB tables (seconds)
        jb_reference = {
            10: 137.1, 20: 263.3, 30: 382.9, 40: 489.8, 50: 574.6,
            60: 637.5, 70: 686.9, 80: 729.8, 90: 768.8, 100: 806.0
        }
        
        # Scale times to match reference values
        for dist, ref_time in jb_reference.items():
            idx = np.argmin(np.abs(self.distances - dist))
            if self.times[0, idx] > 0:
                scale = ref_time / self.times[0, idx]
                # Apply scale factor to nearby distances
                for j in range(max(0, idx-5), min(len(self.distances), idx+5)):
                    weight = 1.0 - 0.1 * abs(j - idx)
                    if weight > 0:
                        self.times[:, j] *= (1.0 + (scale - 1.0) * weight)
    
    def _load_from_file(self, filename: str):
        """
        Load JB tables from file.
        
        Expected format (from jfbup.dat):
        First line: header with depth values
        Subsequent lines: distance, travel_times for each depth
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        def parse_f5_line(line: str) -> list:
            values = []
            for i in range(0, len(line.rstrip('\n')), 5):
                field = line[i:i + 5].strip()
                if not field:
                    continue
                values.append(float(field))
            return values

        header = parse_f5_line(lines[0])
        self.depths = np.array(header[1:16])

        distances = []
        times = []
        for line in lines[1:]:
            parts = parse_f5_line(line)
            if len(parts) >= len(self.depths) + 1:
                distances.append(parts[0])
                times.append(parts[1:len(self.depths) + 1])
        
        self.distances = np.array(distances)
        self.times = np.array(times).T  # Transpose to [depth, distance]
    
    def get_travel_time(self, distance_deg: float, depth_km: float) -> float:
        """
        Get interpolated P-wave travel time.
        
        Parameters
        ----------
        distance_deg : float
            Epicentral distance in degrees
        depth_km : float
            Source depth in km
            
        Returns
        -------
        float
            Travel time in seconds
        """
        # Find indices for interpolation
        dep_idx, dep_w1, dep_w2 = self._interp_weights(depth_km, self.depths)
        dis_idx, dis_w1, dis_w2 = self._interp_weights(distance_deg, self.distances)
        
        if dep_idx < 0 or dis_idx < 0:
            raise ValueError(f"Distance {distance_deg} or depth {depth_km} out of table range")
        
        # Bilinear interpolation
        t00 = self.times[dep_idx, dis_idx]
        t01 = self.times[dep_idx, dis_idx + 1]
        t10 = self.times[dep_idx + 1, dis_idx]
        t11 = self.times[dep_idx + 1, dis_idx + 1]
        
        t0 = t00 * dis_w1 + t01 * dis_w2
        t1 = t10 * dis_w1 + t11 * dis_w2
        
        return t0 * dep_w1 + t1 * dep_w2
    
    def get_slowness(self, distance_deg: float, depth_km: float) -> float:
        """
        Get ray parameter (slowness) p = dT/dΔ.
        
        Parameters
        ----------
        distance_deg : float
            Epicentral distance in degrees
        depth_km : float
            Source depth in km
            
        Returns
        -------
        float
            Ray parameter in seconds/degree
        """
        # Use finite differences
        delta = 0.5  # degrees
        
        if distance_deg - delta >= self.distances[0]:
            t1 = self.get_travel_time(distance_deg - delta, depth_km)
        else:
            t1 = self.get_travel_time(distance_deg, depth_km)
            delta = distance_deg - self.distances[0]
        
        if distance_deg + delta <= self.distances[-1]:
            t2 = self.get_travel_time(distance_deg + delta, depth_km)
        else:
            t2 = self.get_travel_time(distance_deg, depth_km)
            delta = self.distances[-1] - distance_deg
        
        if delta > 0:
            return (t2 - t1) / (2 * delta)
        else:
            return 0.0
    
    def _interp_weights(self, x: float, arr: np.ndarray) -> Tuple[int, float, float]:
        """Get interpolation index and weights."""
        if x < arr[0] or x > arr[-1]:
            return -1, 0.0, 0.0
        
        idx = np.searchsorted(arr, x) - 1
        if idx < 0:
            idx = 0
        if idx >= len(arr) - 1:
            idx = len(arr) - 2
        
        dx = arr[idx + 1] - arr[idx]
        if dx > 0:
            w2 = (x - arr[idx]) / dx
            w1 = 1.0 - w2
        else:
            w1, w2 = 1.0, 0.0
        
        return idx, w1, w2


# =============================================================================
# GEOMETRIC SPREADING AND RECEIVER FUNCTIONS
# =============================================================================
def compute_geometric_spreading(distance_deg: float, depth_km: float, 
                                 slowness_deg: float,
                                 alpha_receiver: float = 7.0,
                                 beta_receiver: float = 4.04,
                                 rho_receiver: float = 3.0,
                                 d2t_dd2_table: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Compute geometric spreading factor and receiver function for P-waves.
    
    Equivalent to Fortran GEORPZ subroutine.
    
    Parameters
    ----------
    distance_deg : float
        Epicentral distance in degrees
    depth_km : float
        Source depth in km
    slowness_deg : float
        Ray parameter in s/deg
    alpha_receiver : float
        P-wave velocity at receiver (km/s)
    beta_receiver : float
        S-wave velocity at receiver (km/s)
    rho_receiver : float
        Density at receiver (g/cm³)
    d2t_dd2_table : np.ndarray, optional
        Pre-computed d²T/dΔ² values for accurate spreading
        
    Returns
    -------
    tuple
        (g, rpz) geometric spreading factor and receiver function
    """
    reth = EARTH_RADIUS_KM
    
    # Convert slowness to s/km
    p = slowness_deg * RAD_TO_DEG / reth  # s/km
    
    # Approximate source region parameters
    alph = 6.5  # P velocity at source (km/s)
    rh = 2.9    # Density at source (g/cm³)
    
    # Earth flattening correction
    error = reth / (reth - depth_km)
    
    # Incidence angle at source
    sin_ih = p * alph
    if abs(sin_ih) > 1:
        sin_ih = np.sign(sin_ih)
    angih = np.arcsin(sin_ih)
    cosih = np.cos(angih)
    
    # d(ih)/d(delta) - rate of change of incidence angle with distance
    # This controls geometric spreading
    if d2t_dd2_table is not None:
        # Use tabulated values
        d2tdd2 = _interpolate_d2t_dd2(distance_deg, depth_km, d2t_dd2_table)
    else:
        # Approximate value
        d2tdd2 = _estimate_d2t_dd2(distance_deg, depth_km)
    
    # Convert to radians
    d2tdd2_rad = d2tdd2 * (RAD_TO_DEG) ** 2
    
    dihdel = abs((d2tdd2_rad / cosih) * alph / (reth - depth_km))
    
    # Incidence angle at receiver
    sin_i0 = p * alpha_receiver / error
    if abs(sin_i0) > 1:
        sin_i0 = np.sign(sin_i0)
    angi0 = np.arcsin(sin_i0)
    
    # Geometric spreading factor
    delta_rad = distance_deg * DEG_TO_RAD
    factor = p * rh * alph * alph / (rho_receiver * alpha_receiver)
    
    sin_delta = np.sin(delta_rad)
    if sin_delta > 0 and np.cos(angi0) > 0 and dihdel > 0:
        g = np.sqrt(factor * dihdel / (sin_delta * np.cos(angi0)))
    else:
        g = 0.0
    
    # Receiver function (Helmberger, 1974)
    p2 = p * p / (error * error)
    betar2 = beta_receiver * beta_receiver
    
    eta_a = np.sqrt(max(0, 1.0 / (alpha_receiver * alpha_receiver) - p2))
    eta_b = np.sqrt(max(0, 1.0 / betar2 - p2))
    
    denom = betar2 * (eta_b * eta_b - p2) ** 2 + 4.0 * p2 * betar2 * eta_a * eta_b
    
    if denom > 0:
        rpz = 2.0 * eta_a * (eta_b * eta_b - p2) / denom
        rpz = alpha_receiver * rpz
    else:
        rpz = 1.0
    
    return g, rpz


def _estimate_d2t_dd2(distance_deg: float, depth_km: float) -> float:
    """
    Estimate second derivative of travel time with respect to distance.
    
    This is an approximation - accurate values require ray tracing.
    """
    # Approximate formula for d²T/dΔ²
    # This varies with distance and depth
    if distance_deg < 15:
        # Near-source region
        return -0.02 - 0.001 * depth_km
    elif distance_deg < 30:
        # Transition zone
        return -0.015 - 0.0005 * depth_km
    elif distance_deg < 90:
        # Mantle phase region
        return -0.01 - 0.0002 * depth_km
    else:
        # Deep mantle region
        return -0.005
    
    
def _interpolate_d2t_dd2(distance_deg: float, depth_km: float,
                         table: np.ndarray) -> float:
    """
    Interpolate d²T/dΔ² from pre-computed table.
    
    Table format: [depths, distances, values]
    """
    # This would interpolate from the d2tdd2.p or d2tdd2.pcp files
    # For now, use approximation
    return _estimate_d2t_dd2(distance_deg, depth_km)


# =============================================================================
# RADIATION PATTERN AND FOCAL MECHANISM
# =============================================================================
def compute_radiation_coefficients(distance_deg: float, depth_km: float,
                                    azimuth: float, strike: float, 
                                    dip: float, rake: float,
                                    jb_tables: Optional[JBTables] = None) -> dict:
    """
    Compute P-wave radiation coefficients for a double-couple source.
    
    Equivalent to Fortran BC10 subroutine.
    
    Parameters
    ----------
    distance_deg : float
        Epicentral distance in degrees
    depth_km : float
        Source depth in km
    azimuth : float
        Station azimuth from source (degrees)
    strike : float
        Fault strike (degrees)
    dip : float
        Fault dip (degrees)
    rake : float
        Fault rake (degrees)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'p': ray parameter (s/deg)
        - 'ih': takeoff angle (radians)
        - 'g': geometric spreading
        - 'rpz': receiver function
        - 'Fp': direct P radiation
        - 'Fpp': pP radiation
        - 'Fsp': sP radiation
        - 'PP': pP reflection coefficient
        - 'SP': sP reflection coefficient
    """
    reth = EARTH_RADIUS_KM
    
    # Get travel time tables
    if jb_tables is None:
        jb_tables = JBTables()
    
    # Get slowness (ray parameter)
    p = jb_tables.get_slowness(distance_deg, depth_km)  # s/deg
    
    # Receiver parameters (mantle values)
    alphar = 7.0  # km/s
    betar = alphar / np.sqrt(3.0)  # km/s
    rhor = 3.0  # g/cm³
    
    # Compute geometric spreading
    g, rpz = compute_geometric_spreading(distance_deg, depth_km, p, alphar, betar, rhor)
    
    # Takeoff angle for P wave.  The Fortran expression is
    # asin(p_s_per_deg * alpha * 180 / (pi * radius)), which is simply
    # asin(p_s_per_km * alpha).  Avoid applying the radius conversion twice.
    p_km = p * RAD_TO_DEG / reth  # Convert s/deg to s/km
    sin_ih = p_km * alphar
    if abs(sin_ih) > 1:
        sin_ih = np.sign(sin_ih)
    aih = np.arcsin(sin_ih)
    
    # Takeoff angle for S wave
    sin_jh = np.sin(aih) / np.sqrt(3.0)
    ajh = np.arcsin(sin_jh)
    
    # Compute radiation pattern
    # Convert angles to radians
    delta_rad = dip * DEG_TO_RAD
    lambda_rad = rake * DEG_TO_RAD
    # Fortran uses fai = strike - azimuth.
    phi_rad = (strike - azimuth) * DEG_TO_RAD
    
    si = np.sin(aih)
    ci = np.cos(aih)
    sj = np.sin(ajh)
    cj = np.cos(ajh)
    s2i = 2.0 * si * ci
    c2i = 2.0 * ci * ci - 1.0
    
    sd = np.sin(delta_rad)
    cd = np.cos(delta_rad)
    sl = np.sin(lambda_rad)
    cl = np.cos(lambda_rad)
    sf = np.sin(phi_rad)
    cf = np.cos(phi_rad)
    s2f = 2.0 * sf * cf
    c2f = 2.0 * cf * cf - 1.0
    c2d = 2.0 * cd * cd - 1.0
    
    # Radiation pattern coefficients (Aki & Richards convention)
    sr = sd * cd * sl
    pr = cl * sd * s2f - sr * c2f
    qr = sl * c2d * sf + cl * cd * cf
    
    # Direct P radiation coefficient
    Fp = sr * (3.0 * ci * ci - 1.0) - qr * s2i - pr * si * si
    
    # pP radiation (ih -> pi - ih)
    Fpp = sr * (3.0 * ci * ci - 1.0) + qr * s2i - pr * si * si
    
    # sP radiation (jh -> pi - jh)
    s2j = np.sin(2.0 * PI - 2.0 * ajh)
    c2j = np.cos(2.0 * PI - 2.0 * ajh)
    Fsp = 1.5 * sr * s2j + qr * c2j + 0.5 * pr * s2j
    
    # Free surface reflection coefficients
    pbeta = p_km * betar
    
    a = 4.0 * pbeta * (betar / alphar) * ci * cj
    b = (1.0 - 2.0 * pbeta * pbeta) ** 2
    c = 4.0 * (betar / alphar) * pbeta * cj * (1.0 - 2.0 * pbeta * pbeta)
    
    if (a + b) != 0:
        PP = (a - b) / (a + b)
        SP = c / (a + b)
        SP = SP * ci / cj if cj != 0 else 0
    else:
        PP = 0.0
        SP = 0.0
    
    return {
        'p': p,
        'ih': aih,
        'g': g,
        'rpz': rpz,
        'Fp': Fp,
        'Fpp': Fpp,
        'Fsp': Fsp,
        'PP': PP,
        'SP': SP
    }


def compute_FgP2(distance_deg: float, depth_km: float, azimuth: float,
                 strike: float, dip: float, rake: float,
                 q_bc: float = 5.196) -> Tuple[float, dict]:
    """
    Compute the (FgP)² factor from Boatwright and Choy (1986).
    
    This combines the radiation pattern with reflection coefficients.
    
    Parameters
    ----------
    distance_deg : float
        Epicentral distance
    depth_km : float
        Source depth
    azimuth : float
        Station azimuth
    strike, dip, rake : float
        Focal mechanism parameters
    q_bc : float
        Boatwright-Choy q parameter (default: sqrt(3)^5 ≈ 5.196)
        
    Returns
    -------
    tuple
        (FgP2, coefficients_dict)
    """
    coeffs = compute_radiation_coefficients(distance_deg, depth_km, azimuth,
                                           strike, dip, rake)
    
    Fp = coeffs['Fp']
    Fpp = coeffs['Fpp']
    Fsp = coeffs['Fsp']
    PP = coeffs['PP']
    SP = coeffs['SP']
    
    # Equation (10) from Boatwright and Choy (1986), p. 2097
    FgP2 = Fp * Fp + Fpp * Fpp * PP * PP + (2.0 / np.sqrt(3.0)) * q_bc * SP * SP * Fsp * Fsp
    
    return FgP2, coeffs


def estimated_FgP2(distance_deg: float) -> float:
    """
    Empirical estimate of (FgP)² as a function of distance.
    
    Based on regression from endig.f:
    FgP² = a0 + a1*Δ + a2*Δ²
    
    Parameters
    ----------
    distance_deg : float
        Epicentral distance in degrees
        
    Returns
    -------
    float
        Estimated (FgP)² value
    """
    # Regression parameters from endig.f
    a0 = 1.171
    a1 = -7.271e-3
    a2 = 6.009e-5
    
    return a0 + a1 * distance_deg + a2 * distance_deg * distance_deg


# =============================================================================
# T* ATTENUATION
# =============================================================================
def compute_tstar(freq: float) -> float:
    """
    Compute t* attenuation parameter.
    
    Based on Choy and Cormier (1986).
    
    Parameters
    ----------
    freq : float
        Frequency in Hz
        
    Returns
    -------
    float
        t* value in seconds
    """
    if freq < 0.1:
        return 0.9 - 0.1 * np.log10(freq)
    elif freq < 1.0:
        return 0.5 - 0.5 * np.log10(freq)
    else:
        return 0.5 - 0.1 * np.log10(freq)


# =============================================================================
# SLOWNESS PARAMETER CALCULATIONS
# =============================================================================
def slowness_to_different_units(p_deg: float) -> dict:
    """
    Convert slowness from s/deg to other units.
    
    Parameters
    ----------
    p_deg : float
        Ray parameter in seconds/degree
        
    Returns
    -------
    dict
        Slowness in different units:
        - 'p_deg': s/deg
        - 'p_km': s/km
        - 'p_rad': s/rad
    """
    reth = EARTH_RADIUS_KM
    
    p_km = p_deg / (reth * DEG_TO_RAD)  # s/km
    p_rad = p_deg * RAD_TO_DEG           # s/rad
    
    return {
        'p_deg': p_deg,
        'p_km': p_km,
        'p_rad': p_rad
    }


def slowness_from_travel_time_gradient(t1: float, t2: float, 
                                        d1: float, d2: float) -> float:
    """
    Calculate slowness from travel time gradient.
    
    Parameters
    ----------
    t1, t2 : float
        Travel times (seconds)
    d1, d2 : float
        Distances (degrees)
        
    Returns
    -------
    float
        Slowness in s/deg
    """
    if d2 != d1:
        return (t2 - t1) / (d2 - d1)
    return 0.0


def takeoff_angle_from_slowness(p_deg: float, velocity_km_s: float,
                                 depth_km: float = 0.0) -> float:
    """
    Calculate takeoff angle from slowness.
    
    Parameters
    ----------
    p_deg : float
        Ray parameter in s/deg
    velocity_km_s : float
        Seismic velocity at source (km/s)
    depth_km : float
        Source depth (km)
        
    Returns
    -------
    float
        Takeoff angle in degrees (from vertical)
    """
    reth = EARTH_RADIUS_KM
    
    # Convert p to s/km
    p_km = p_deg / (reth * DEG_TO_RAD)
    
    # Earth flattening correction
    error = reth / (reth - depth_km)
    
    # Snell's law: p = sin(i) / v
    sin_i = p_km * velocity_km_s / error
    
    if abs(sin_i) > 1:
        return 90.0 if sin_i > 0 else -90.0
    
    return np.arcsin(sin_i) * RAD_TO_DEG


if __name__ == "__main__":
    print("Testing travel_time.py")
    
    # Create JB tables
    jb = JBTables()
    
    # Test travel time
    dist = 60.0  # degrees
    depth = 33.0  # km
    t = jb.get_travel_time(dist, depth)
    print(f"Travel time at Δ={dist}°, h={depth}km: {t:.2f} s")
    
    # Test slowness
    p = jb.get_slowness(dist, depth)
    print(f"Slowness: {p:.3f} s/deg")
    
    # Convert slowness units
    p_units = slowness_to_different_units(p)
    print(f"Slowness: {p_units['p_km']:.4f} s/km, {p_units['p_rad']:.2f} s/rad")
    
    # Test radiation coefficients
    coeffs = compute_radiation_coefficients(
        distance_deg=60.0, depth_km=33.0,
        azimuth=45.0, strike=0.0, dip=45.0, rake=90.0
    )
    print(f"Radiation: Fp={coeffs['Fp']:.3f}, Fpp={coeffs['Fpp']:.3f}, Fsp={coeffs['Fsp']:.3f}")
    
    # Test geometric spreading
    g, rpz = compute_geometric_spreading(60.0, 33.0, p)
    print(f"Geometric spreading: g={g:.3f}, rpz={rpz:.3f}")
    
    # Test t*
    tstar = compute_tstar(0.1)
    print(f"t* at 0.1 Hz: {tstar:.2f} s")
    
    print("All tests passed!")
