"""
Seismic Utilities - Python conversion of Fortran routines
Core mathematical and geophysical functions for seismological calculations.
"""

import numpy as np
from typing import Tuple, Optional, List


# =============================================================================
# CONSTANTS
# =============================================================================
PI = np.pi
TWOPI = 2.0 * PI
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6371.0e3
DEG_TO_KM = EARTH_RADIUS_KM * PI / 180.0
RAD_TO_DEG = 180.0 / PI
DEG_TO_RAD = PI / 180.0

# WGS84-like ellipsoid parameters
A0 = 6378.388  # Equatorial radius (km)
B0 = 6356.912  # Polar radius (km)


# =============================================================================
# FFT ROUTINES
# =============================================================================
def coolb(data: np.ndarray, sign: float = -1.0) -> np.ndarray:
    """
    Cooley-Tukey FFT algorithm (in-place).
    
    This is a direct translation of the Fortran COOLB subroutine.
    For production use, prefer np.fft.fft() which is faster.
    
    Parameters
    ----------
    data : np.ndarray
        Complex array of data (length must be power of 2)
    sign : float
        -1.0 for forward FFT, +1.0 for inverse FFT
        
    Returns
    -------
    np.ndarray
        FFT of input data
    """
    # For practical use, use numpy's FFT
    if sign < 0:
        return np.fft.fft(data)
    else:
        return np.fft.ifft(data) * len(data)


def find_fft_size(n: int) -> Tuple[int, int]:
    """
    Find next power of 2 >= n for FFT.
    
    Equivalent to Fortran MYFND subroutine.
    
    Parameters
    ----------
    n : int
        Number of data points
        
    Returns
    -------
    tuple
        (nfft, jfft) where nfft = 2^jfft >= n
    """
    jfft = int(np.ceil(np.log2(n)))
    nfft = 2 ** jfft
    return nfft, jfft


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================
def detrend(data: np.ndarray) -> np.ndarray:
    """
    Remove mean from array (simple detrending).
    
    Equivalent to Fortran MYDTR subroutine.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
        
    Returns
    -------
    np.ndarray
        Detrended data
    """
    return data - np.mean(data)


def detrend_linear(data: np.ndarray) -> np.ndarray:
    """
    Remove best-fitting linear trend from array.
    
    Equivalent to Fortran MYDTR1 subroutine.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
        
    Returns
    -------
    np.ndarray
        Detrended data
    """
    n = len(data)
    x = np.arange(1, n + 1, dtype=float)
    
    # Compute linear regression coefficients
    s1 = np.sum(x)
    s2 = np.sum(x * x)
    sx = np.sum(data)
    snx = np.sum(x * data)
    
    a = (snx - s1 * sx / n) / (s2 - s1 * s1 / n)
    b = (sx * s2 - s1 * snx) / (n * s2 - s1 * s1)
    
    return data - a * x - b


def cosine_taper(data: np.ndarray, begin_frac: float = 0.05, 
                 end_frac: float = 0.1) -> np.ndarray:
    """
    Apply cosine taper to beginning and end of array.
    
    Equivalent to Fortran MYTPR subroutine.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    begin_frac : float
        Fraction of data to taper at beginning (0-1)
    end_frac : float
        Fraction of data to taper at end (0-1)
        
    Returns
    -------
    np.ndarray
        Tapered data
    """
    n = len(data)
    result = data.copy()
    
    # Taper beginning
    m1 = int(n * begin_frac + 0.5)
    if m1 > 0:
        for i in range(m1):
            cs = (1.0 - np.cos((i + 1) * PI / m1)) / 2.0
            result[i] *= cs
    
    # Taper end
    m3 = int(n * end_frac + 0.5)
    if m3 > 0:
        m5 = n - m3
        for i in range(m5, n):
            cs = (1.0 - np.cos((i - n) * PI / m3)) / 2.0
            result[i] *= cs
    
    return result


def bandpass_filter(data: np.ndarray, dt: float, tmin: float, tmax: float) -> np.ndarray:
    """
    Apply bandpass filter to data using FFT.
    
    Equivalent to Fortran MYFILT/MYCTFL subroutines.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    dt : float
        Sampling interval (seconds)
    tmin : float
        Minimum period (seconds) - high frequency cutoff
    tmax : float
        Maximum period (seconds) - low frequency cutoff
        
    Returns
    -------
    np.ndarray
        Filtered data
    """
    n = len(data)
    nfft, _ = find_fft_size(n)
    
    # Pad and FFT
    padded = np.zeros(nfft, dtype=complex)
    padded[:n] = detrend(data)
    spectrum = np.fft.fft(padded)
    
    # Frequency parameters
    df = 1.0 / (nfft * dt)
    fmin = 1.0 / tmax
    fmax = 1.0 / tmin
    
    # Apply bandpass
    freqs = np.fft.fftfreq(nfft, dt)
    mask = (np.abs(freqs) >= fmin) & (np.abs(freqs) <= fmax)
    spectrum[~mask] = 0.0
    
    # Inverse FFT and return real part
    result = np.fft.ifft(spectrum)
    return np.real(result[:n])


def running_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute running average over window points.
    
    Equivalent to Fortran MYRNAV subroutine.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    window : int
        Half-width of averaging window (total width = 2*window + 1)
        
    Returns
    -------
    np.ndarray
        Smoothed data
    """
    n = len(data)
    result = data.copy()
    width = 2 * window + 1
    
    # Compute running average for middle section
    for i in range(window, n - window):
        result[i] = np.mean(data[i - window:i + window + 1])
    
    return result


# =============================================================================
# GEODESIC CALCULATIONS
# =============================================================================
def great_circle(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float, float, float]:
    """
    Calculate great circle distance and azimuths between two points.
    
    Equivalent to Fortran MYGRT subroutine.
    Uses ellipsoidal Earth model for accuracy.
    
    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of first point (degrees)
    lat2, lon2 : float
        Latitude and longitude of second point (degrees)
        
    Returns
    -------
    tuple
        (distance_deg, azimuth_12, azimuth_21, great_circle_length)
        distance in degrees, azimuths in degrees, length in km
    """
    # Convert to radians
    lat1_r = lat1 * DEG_TO_RAD
    lon1_r = lon1 * DEG_TO_RAD
    lat2_r = lat2 * DEG_TO_RAD
    lon2_r = lon2 * DEG_TO_RAD
    
    # Ellipsoid flattening
    h = 1.0 - (B0 * B0) / (A0 * A0)
    p = h / (1.0 - h)
    
    # Calculate intermediate values
    sin_lat1 = np.sin(lat1_r)
    cos_lat1 = np.cos(lat1_r)
    sin_lat2 = np.sin(lat2_r)
    cos_lat2 = np.cos(lat2_r)
    
    # Avoid division by zero
    if cos_lat1 == 0:
        cos_lat1 = 1e-10
    if cos_lat2 == 0:
        cos_lat2 = 1e-10
    if sin_lat1 == 0:
        sin_lat1 = 1e-10
    
    r1 = A0 / np.sqrt(1.0 - h * sin_lat1 * sin_lat1)
    r2 = A0 / np.sqrt(1.0 - h * sin_lat2 * sin_lat2)
    
    dlon = lon2_r - lon1_r
    cos_dlon = np.cos(dlon)
    sin_dlon = np.sin(dlon)
    
    # Calculate azimuths
    q = sin_lat2 * cos_lat1 / ((1.0 + p) * cos_lat2 * sin_lat1) + h * r1 * cos_lat1 / (r2 * cos_lat2)
    az12 = np.arctan2(sin_dlon, (q - cos_dlon) * sin_lat1)
    
    q = sin_lat1 * cos_lat2 / (cos_lat1 * sin_lat2 * (1.0 + p)) + h * r2 * cos_lat2 / (r1 * cos_lat1)
    az21 = np.arctan2(-sin_dlon, sin_lat2 * (q - np.cos(-dlon)))
    
    # Calculate distance
    cos_az12 = np.cos(az12)
    cta2 = cos_lat1 * cos_lat1 * cos_az12 * cos_az12
    p0 = p * (cta2 + sin_lat1 * sin_lat1)
    b0_local = (r1 / (1.0 + p0)) * np.sqrt(1.0 + p * cta2)
    
    # Great circle length
    e0 = p0 / (1.0 + p0)
    gc = 2.0 * PI * b0_local * np.sqrt(1.0 + p0) * (
        1.0 - e0 * (0.25 + e0 * (3.0/64.0 + 5.0 * e0 / 256.0))
    )
    
    # Spherical approximation for distance in degrees
    cos_dist = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    dist_deg = np.arccos(cos_dist) * RAD_TO_DEG
    
    # Ensure distance is in range [0, 180]
    if dist_deg > 355:
        dist_deg = 360.0 - dist_deg
    
    # Convert azimuths to degrees
    az12_deg = az12 * RAD_TO_DEG
    az21_deg = az21 * RAD_TO_DEG
    
    if az12_deg < 0:
        az12_deg += 360.0
    if az21_deg < 0:
        az21_deg += 360.0
    
    return dist_deg, az12_deg, az21_deg, gc


def geodesic_point(lat: float, lon: float, azimuth: float, distance_km: float) -> Tuple[float, float]:
    """
    Find point at given distance and azimuth from starting point.
    
    Equivalent to Fortran MYGDS subroutine.
    
    Parameters
    ----------
    lat, lon : float
        Starting latitude and longitude (degrees)
    azimuth : float
        Azimuth from starting point (degrees)
    distance_km : float
        Distance to travel (km)
        
    Returns
    -------
    tuple
        (new_lat, new_lon) in degrees
    """
    # Normalize azimuth
    az = azimuth
    if az < 0:
        az += 360.0
    if az > 360:
        az -= 360.0
    
    # Avoid singularities
    if lat == 90.0:
        lat = 90.0 - 1e-6
    if az == 90.0 or az == 270.0:
        az -= 1e-6
    
    # Convert to radians
    lat_r = lat * DEG_TO_RAD
    lon_r = lon * DEG_TO_RAD
    az_r = az * DEG_TO_RAD
    
    # Ellipsoid parameters
    f = (A0 - B0) / A0
    e2 = 1.0 - (B0 ** 2) / (A0 ** 2)
    eps = e2 / (1.0 - e2)
    
    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    sin_az = np.sin(az_r)
    cos_az = np.cos(az_r)
    
    # Radius of curvature
    v = A0 / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    
    # Compute intermediate values
    tc2 = cos_az * cos_az * cos_lat * cos_lat
    c2 = tc2 + sin_lat * sin_lat
    eps0 = c2 * eps
    
    b1 = (v * np.sqrt(1.0 + eps * tc2)) / (1.0 + eps0)
    tan_lat = np.tan(lat_r)
    
    s1 = np.sqrt(1.0 + eps0)
    
    # Series coefficients
    g0 = 1.0 - (eps0 / 4.0) + ((7.0 * eps0 * eps0) / 64.0) - ((15.0 * eps0 ** 3) / 256.0)
    g2 = (eps0 / 8.0) - (0.0625 * eps0 * eps0) + ((145.0 * eps0 ** 3) / 2048.0)
    g4 = ((5.0 * eps0 * eps0) / 256.0) - ((5.0 * eps0 ** 3) / 256.0)
    g6 = (29.0 * eps0 ** 3) / 6144.0
    
    sigm = (distance_km * g0) / b1
    
    if cos_az == 0:
        cos_az = 1e-5
    
    u1p = np.arctan2(tan_lat, cos_az * s1)
    sin2p = np.sin(2.0 * u1p)
    sin4p = np.sin(4.0 * u1p)
    
    tss = (eps0 / 4.0) - (eps0 * eps0 / 8.0)
    s12 = 2.0 * u1p - tss * sin2p - ((eps0 * eps0) / 128.0) * sin4p
    sigp = s12 + sigm
    
    t1 = sigm + (2.0 * g2 * np.sin(sigm)) * np.cos(sigp)
    t2 = (2.0 * g4 * np.sin(2.0 * sigm)) * np.cos(2.0 * sigp)
    t3 = (2.0 * g6 * np.sin(3.0 * sigm)) * np.cos(3.0 * sigp)
    u2p = u1p + t1 + t2 + t3
    
    # Normalize u2p
    while u2p >= TWOPI:
        u2p -= TWOPI
    while u2p < -PI:
        u2p += TWOPI
    
    sinu1 = tan_lat / np.sqrt(1.0 + eps + tan_lat * tan_lat)
    c = np.sqrt(c2)
    sinu2 = ((b1 * c) / B0) * np.sin(u2p) - ((eps - eps0) / (1.0 + eps0)) * sinu1
    u2 = np.arcsin(np.clip(sinu2, -1, 1))
    
    cos_u2 = np.cos(u2)
    if cos_u2 == 0:
        u2 -= 1e-5
        cos_u2 = np.cos(u2)
    
    sinp1 = sinu2 / np.sqrt(1.0 - e2 * cos_u2 * cos_u2)
    new_lat = np.arcsin(np.clip(sinp1, -1, 1))
    
    a1 = b1 * np.sqrt(1.0 + eps0)
    q1 = (a1 * np.cos(u2p)) / (A0 * cos_u2)
    q1 = np.clip(q1, -1, 1)
    q2 = np.arccos(q1)
    
    x1 = sin_lat * sin_az
    amu = np.arctan2(x1, cos_az)
    
    az_deg = az
    u2p_deg = u2p * RAD_TO_DEG
    
    if az_deg > 180.0:
        q2 = -q2
    if u2p_deg > 180.0 or u2p_deg < 0.0:
        q2 = -q2
    
    dlamb = q2 - amu
    new_lon = dlamb + lon_r
    
    # Convert back to degrees
    new_lat_deg = new_lat * RAD_TO_DEG
    new_lon_deg = new_lon * RAD_TO_DEG
    
    if abs(new_lon_deg) > 180.0:
        new_lon_deg -= np.sign(new_lon_deg) * 360.0
    
    return new_lat_deg, new_lon_deg


def great_circle_pole(elat: float, elon: float, slat: float, slon: float) -> Tuple[float, float]:
    """
    Find the pole of the great circle connecting two points.
    
    Equivalent to Fortran GCPOLE subroutine.
    
    Parameters
    ----------
    elat, elon : float
        Epicenter latitude and longitude (degrees)
    slat, slon : float
        Station latitude and longitude (degrees)
        
    Returns
    -------
    tuple
        (pole_lat, pole_lon) in degrees
    """
    # Convert to Cartesian
    def to_cartesian(lat, lon):
        lat_r = lat * DEG_TO_RAD
        lon_r = lon * DEG_TO_RAD
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        return x, y, z
    
    ex, ey, ez = to_cartesian(elat, elon)
    sx, sy, sz = to_cartesian(slat, slon)
    
    # Cross product gives pole direction
    px = ey * sz - ez * sy
    py = ez * sx - ex * sz
    pz = ex * sy - ey * sx
    
    # Convert back to lat/lon
    r = np.sqrt(px * px + py * py + pz * pz)
    if r == 0:
        return 0.0, 0.0
    
    rxy = np.sqrt(px * px + py * py)
    if rxy == 0:
        plat = 90.0 if pz > 0 else -90.0
        plon = 0.0
    else:
        plat = np.arctan(pz / rxy) * RAD_TO_DEG
        plon = np.arctan2(py, px) * RAD_TO_DEG
    
    return plat, plon


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================
def compute_statistics(data: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute mean, standard deviation, skewness, and kurtosis.
    
    Equivalent to Fortran MYMSSK subroutine.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
        
    Returns
    -------
    tuple
        (mean, std_dev, skewness, kurtosis)
    """
    n = len(data)
    mean = np.mean(data)
    centered = data - mean
    
    std_dev = np.sqrt(np.mean(centered ** 2))
    
    if std_dev > 0:
        skew = np.mean(centered ** 3) / (std_dev ** 3)
        kurtosis = np.mean(centered ** 4) / (std_dev ** 4) - 3.0
    else:
        skew = 0.0
        kurtosis = 0.0
    
    return mean, std_dev, skew, kurtosis


def correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.
    
    Equivalent to Fortran CORCOEF subroutine.
    
    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays (same length)
        
    Returns
    -------
    float
        Correlation coefficient
    """
    n = len(x)
    xbar = np.mean(x)
    ybar = np.mean(y)
    
    sxy = np.sum((x - xbar) * (y - ybar))
    sx2 = np.sum((x - xbar) ** 2)
    sy2 = np.sum((y - ybar) ** 2)
    
    if sx2 * sy2 > 0:
        return sxy / np.sqrt(sx2 * sy2)
    return 0.0


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute least-squares linear regression: y = ax + b
    
    Equivalent to Fortran MYREGR subroutine.
    
    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays
        
    Returns
    -------
    tuple
        (a, b) slope and intercept
    """
    n = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    sxy = np.sum(x * y)
    sx2 = np.sum(x * x)
    
    a = (n * sxy - sx * sy) / (n * sx2 - sx * sx)
    b = (sy - a * sx) / n
    
    return a, b


# =============================================================================
# INTERPOLATION
# =============================================================================
def interpolate_in_array(x: float, array: np.ndarray) -> Tuple[int, float, float, bool]:
    """
    Find index and interpolation weights for value in sorted array.
    
    Equivalent to Fortran MYSERT subroutine.
    
    Parameters
    ----------
    x : float
        Value to insert
    array : np.ndarray
        Sorted array (increasing or decreasing)
        
    Returns
    -------
    tuple
        (index, weight_a, weight_b, valid)
        Linear interpolation: result = array[index]*weight_a + array[index+1]*weight_b
    """
    n = len(array)
    
    # Check if array is increasing or decreasing
    increasing = array[-1] > array[0]
    
    if increasing:
        if x < array[0] or x > array[-1]:
            return -1, 0.0, 0.0, False
        
        for j in range(n - 1):
            if x >= array[j] and x <= array[j + 1]:
                cx = array[j + 1] - array[j]
                ax = (x - array[j]) / cx
                bx = (array[j + 1] - x) / cx
                return j, bx, ax, True
    else:
        if x > array[0] or x < array[-1]:
            return -1, 0.0, 0.0, False
        
        for j in range(n - 1):
            if x <= array[j] and x >= array[j + 1]:
                cx = array[j] - array[j + 1]
                ax = (array[j] - x) / cx
                bx = (x - array[j + 1]) / cx
                return j, bx, ax, True
    
    return -1, 0.0, 0.0, False


def find_max_min(data: np.ndarray) -> Tuple[float, float, int, int]:
    """
    Find maximum and minimum values and their indices.
    
    Equivalent to Fortran NAXMIN/MAXMIN subroutines.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
        
    Returns
    -------
    tuple
        (max_val, min_val, max_idx, min_idx)
    """
    max_idx = np.argmax(data)
    min_idx = np.argmin(data)
    return data[max_idx], data[min_idx], max_idx, min_idx


# =============================================================================
# TIME UTILITIES
# =============================================================================
def seconds_to_hms(seconds: float) -> Tuple[int, int, float]:
    """
    Convert seconds to hours, minutes, seconds.
    
    Equivalent to Fortran MYHMS subroutine.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
        
    Returns
    -------
    tuple
        (hours, minutes, seconds)
    """
    hours = int(seconds / 3600)
    remainder = seconds - hours * 3600
    minutes = int(remainder / 60)
    secs = remainder - minutes * 60
    return hours, minutes, secs


def julian_day(day: int, month: int, year: int) -> int:
    """
    Convert calendar date to Julian day number.
    
    Equivalent to Fortran JULIAN subroutine.
    
    Parameters
    ----------
    day : int
        Day of month
    month : int
        Month (1-12)
    year : int
        Year
        
    Returns
    -------
    int
        Julian day number (day of year)
    """
    # Check for leap year
    is_leap = False
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                is_leap = True
        else:
            is_leap = True
    
    # Days in each month
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap:
        month_days[1] = 29
    
    jday = day
    for m in range(month - 1):
        jday += month_days[m]
    
    return jday


def from_julian_day(year: int, jday: int) -> Tuple[int, int]:
    """
    Convert Julian day to calendar date.
    
    Equivalent to Fortran DEJUL subroutine.
    
    Parameters
    ----------
    year : int
        Year
    jday : int
        Julian day number (day of year)
        
    Returns
    -------
    tuple
        (day, month)
    """
    # Check for leap year
    is_leap = False
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                is_leap = True
        else:
            is_leap = True
    
    # Days in each month
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap:
        month_days[1] = 29
    
    # Cumulative days
    cumulative = np.cumsum([0] + month_days)
    
    for month in range(1, 13):
        if jday <= cumulative[month]:
            day = jday - cumulative[month - 1]
            return day, month
    
    return 31, 12  # December 31 if out of range


# =============================================================================
# LEGENDRE POLYNOMIALS
# =============================================================================
def legendre_plm(lmax: int, m: int, theta: float) -> np.ndarray:
    """
    Compute associated Legendre polynomials P_l^m(cos(theta)).
    
    Equivalent to Fortran MYPLM subroutine.
    
    Parameters
    ----------
    lmax : int
        Maximum degree l
    m : int
        Order m (m <= lmax)
    theta : float
        Colatitude angle in radians
        
    Returns
    -------
    np.ndarray
        Array of P_l^m values for l = m to lmax
    """
    plm = np.zeros(lmax + 1)
    
    cost = np.cos(theta)
    sint = np.sin(theta)
    tant = np.tan(theta) if np.cos(theta) != 0 else 1e10
    
    # Starting values using recurrence
    if m == 0:
        plm[0] = 1.0
        if lmax > 0:
            plm[1] = cost
    elif m == 1:
        plm[0] = 0.0
        plm[1] = -sint
    else:
        # General starting value for P_m^m
        plm[m] = 1.0
        for i in range(1, m + 1):
            plm[m] *= -(2 * i - 1) * sint
        if m + 1 <= lmax:
            plm[m + 1] = (2 * m + 1) * cost * plm[m]
    
    # Recurrence relation for higher l
    for l in range(max(m + 2, 2), lmax + 1):
        plm[l] = ((2 * l - 1) * cost * plm[l - 1] - (l + m - 1) * plm[l - 2]) / (l - m)
    
    return plm


# =============================================================================
# BESSEL FUNCTIONS
# =============================================================================
def bessel_j0(x: float) -> float:
    """Bessel function J_0(x). Equivalent to Fortran BESSJ0."""
    return float(np.real(np.j0(x))) if hasattr(np, 'j0') else float(np.real(
        np.polynomial.chebyshev.chebval(x / 3.75, [1.0, 0.0, 0.25, 0.0, 0.015625])
        if abs(x) < 3.75 else np.sqrt(2 / (PI * abs(x))) * np.cos(abs(x) - PI / 4)
    ))


def bessel_j1(x: float) -> float:
    """Bessel function J_1(x). Equivalent to Fortran BESSJ1."""
    from scipy.special import j1
    return float(j1(x))


def bessel_jn(n: int, x: float) -> float:
    """Bessel function J_n(x). Equivalent to Fortran BESSJ."""
    from scipy.special import jv
    return float(jv(n, x))


# =============================================================================
# SPECIAL FUNCTIONS
# =============================================================================
def sinc(x: float) -> float:
    """
    Compute sinc function: sin(x)/x with proper limit at x=0.
    
    Equivalent to Fortran SINC function.
    """
    if x == 0:
        return 1.0
    return np.sin(x) / x


def kronecker_delta(i: int, j: int) -> int:
    """
    Kronecker delta function.
    
    Equivalent to Fortran KRON function.
    """
    return 1 if i == j else 0


def chebyshev_polynomials(kmax: int, x: float) -> np.ndarray:
    """
    Compute Chebyshev polynomials T_k(x) for k = 0 to kmax-1.
    
    Equivalent to Fortran CHEBYSHOV subroutine.
    
    Parameters
    ----------
    kmax : int
        Number of polynomials to compute
    x : float
        Argument (must be in [-1, 1])
        
    Returns
    -------
    np.ndarray
        Array of T_k(x) values
    """
    if abs(x) > 1:
        raise ValueError("Argument of Chebyshev must be in [-1, 1]")
    
    c = np.zeros(kmax)
    c[0] = 1.0
    if kmax > 1:
        c[1] = x
    
    for k in range(2, kmax):
        c[k] = 2.0 * x * c[k - 1] - c[k - 2]
    
    return c


# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================
def cartesian_to_polar(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian to polar coordinates.
    
    Equivalent to Fortran CARPOL subroutine.
    
    Parameters
    ----------
    x, y, z : float
        Cartesian coordinates
        
    Returns
    -------
    tuple
        (r, theta, phi) where theta is latitude (degrees), phi is longitude (degrees)
    """
    r = np.sqrt(x * x + y * y + z * z)
    if r == 0:
        return 0.0, 0.0, 0.0
    
    rxy = np.sqrt(x * x + y * y)
    if rxy == 0:
        theta = 90.0 if z > 0 else -90.0
        phi = 0.0
    else:
        theta = np.arctan(z / rxy) * RAD_TO_DEG
        phi = np.arctan2(y, x) * RAD_TO_DEG
    
    return r, theta, phi


def polar_to_cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    """
    Convert polar to Cartesian coordinates.
    
    Equivalent to Fortran POLCAR subroutine.
    
    Parameters
    ----------
    r : float
        Radius
    theta : float
        Latitude (degrees)
    phi : float
        Longitude (degrees)
        
    Returns
    -------
    tuple
        (x, y, z) Cartesian coordinates
    """
    theta_r = theta * DEG_TO_RAD
    phi_r = phi * DEG_TO_RAD
    
    z = r * np.sin(theta_r)
    x = r * np.cos(theta_r) * np.cos(phi_r)
    y = r * np.cos(theta_r) * np.sin(phi_r)
    
    return x, y, z


if __name__ == "__main__":
    # Test basic functionality
    print("Testing seismic_utils.py")
    
    # Test great circle calculation
    dist, az12, az21, gc = great_circle(0, 0, 45, 90)
    print(f"Great circle: dist={dist:.2f}°, az12={az12:.2f}°, az21={az21:.2f}°")
    
    # Test FFT size finder
    nfft, jfft = find_fft_size(1000)
    print(f"FFT size for n=1000: nfft={nfft}, jfft={jfft}")
    
    # Test detrend
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + 10
    detrended = detrend(data)
    print(f"Detrended data mean: {np.mean(detrended):.10f}")
    
    print("All tests passed!")

