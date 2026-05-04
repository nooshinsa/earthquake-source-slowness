"""
Instrument Response Module - Python conversion of Fortran routines
Handles seismometer poles, zeros, and frequency response calculations.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


PI = np.pi
TWOPI = 2.0 * PI


@dataclass
class InstrumentResponse:
    """
    Seismometer instrument response parameters.
    
    Attributes
    ----------
    a0 : float
        Normalization factor
    sensitivity : float
        Sensitivity/gain factor
    zeros : np.ndarray
        Complex zeros array
    poles : np.ndarray
        Complex poles array
    unit : str
        Output unit ('M' for displacement, 'M/S' for velocity, 'M/S**2' for acceleration)
    """
    a0: float
    sensitivity: float
    zeros: np.ndarray
    poles: np.ndarray
    unit: str = 'M'
    
    @property
    def total_gain(self) -> float:
        """Total instrument gain (a0 * sensitivity)."""
        return self.a0 * self.sensitivity
    
    @property
    def n_zeros(self) -> int:
        """Number of zeros."""
        return len(self.zeros)
    
    @property
    def n_poles(self) -> int:
        """Number of poles."""
        return len(self.poles)


def compute_response(freq: float, zeros: np.ndarray, poles: np.ndarray, 
                     gain: float) -> complex:
    """
    Compute instrument response at a single frequency.
    
    Equivalent to Fortran INSTRUMENT subroutine from endig.f.
    
    Parameters
    ----------
    freq : float
        Frequency in Hz
    zeros : np.ndarray
        Complex array of zeros
    poles : np.ndarray
        Complex array of poles
    gain : float
        Total instrument gain
        
    Returns
    -------
    complex
        Complex response at the given frequency
    """
    omega = 2.0 * PI * freq
    z_omega = complex(0, omega)
    
    # Scale to avoid overflow
    log_gain = np.log10(gain) if gain > 0 else 0
    i_exp_g = int(log_gain)
    scaled_gain = gain / (10 ** i_exp_g)
    
    i_exp = i_exp_g
    
    # Numerator (zeros)
    z_num = complex(1, 0)
    for zero in zeros:
        za = z_omega - zero
        if abs(za) > 0:
            i_exp_z = int(np.log10(abs(za)))
            za = za / (10 ** i_exp_z)
            z_num *= za
            i_exp += i_exp_z
    
    # Denominator (poles)
    z_denom = complex(1, 0)
    for pole in poles:
        zb = z_omega - pole
        if abs(zb) > 0:
            i_exp_p = int(np.log10(abs(zb)))
            zb = zb / (10 ** i_exp_p)
            z_denom *= zb
            i_exp -= i_exp_p
    
    # Compute response
    z_res = scaled_gain * z_num / z_denom
    z_res *= (10 ** i_exp)
    
    return z_res


def compute_response_array(freqs: np.ndarray, zeros: np.ndarray, poles: np.ndarray,
                           gain: float) -> np.ndarray:
    """
    Compute instrument response at multiple frequencies.
    
    Parameters
    ----------
    freqs : np.ndarray
        Array of frequencies in Hz
    zeros : np.ndarray
        Complex array of zeros
    poles : np.ndarray
        Complex array of poles
    gain : float
        Total instrument gain
        
    Returns
    -------
    np.ndarray
        Complex response array
    """
    response = np.zeros(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        if f > 0:
            response[i] = compute_response(f, zeros, poles, gain)
    return response


# =============================================================================
# WWSSN and other standard instrument responses
# =============================================================================
def wwssn_lp_response(freq: float, tp: float = 15.0, tg: float = 100.0, 
                      coupling: float = 0.5, hp: float = 1.0, 
                      hg: float = 1.0) -> Tuple[float, float, float]:
    """
    WWSSN Long-period seismograph response.
    
    Equivalent to Fortran MYWWSS subroutine.
    
    Parameters
    ----------
    freq : float
        Frequency in Hz
    tp : float
        Pendulum period (seconds)
    tg : float
        Galvanometer period (seconds)
    coupling : float
        Coupling constant
    hp : float
        Pendulum damping
    hg : float
        Galvanometer damping
        
    Returns
    -------
    tuple
        (amplitude, real_part, imag_part) normalized response
    """
    twopi = 2.0 * PI
    
    fn1 = twopi / tp
    fn2 = twopi / tg
    fk1 = fn1 * hp
    fk2 = fn2 * hg
    fn1s = fn1 * fn1
    fn2s = fn2 * fn2
    
    c1 = fn1s + fn2s + 4.0 * fk1 * fk2 * (1.0 - coupling)
    c2 = fn1s * fn2s
    c3 = 2.0 * (fk1 + fk2)
    c4 = 2.0 * (fk1 * fn2s + fk2 * fn1s)
    
    omega = twopi * freq
    oo = omega * omega
    ooo = omega * oo
    oooo = oo * oo
    
    aim = oooo - c1 * oo + c2
    are = -c3 * ooo + c4 * omega
    sqsq = np.sqrt(aim * aim + are * are)
    
    r = ooo / sqsq if sqsq > 0 else 0
    are = are / sqsq if sqsq > 0 else 0
    aim = aim / sqsq if sqsq > 0 else 0
    
    return r, are, aim


def mechanical_seismograph_response(freq: float, period: float, magnification: float,
                                    damping: float, overdamped: bool = False) -> Tuple[float, float, float]:
    """
    Response of a mechanical seismograph (pendulum).
    
    Equivalent to Fortran MYRSMS subroutine.
    
    Parameters
    ----------
    freq : float
        Frequency in Hz
    period : float
        Natural period of pendulum (seconds)
    magnification : float
        Static magnification
    damping : float
        Damping constant (epsilon or h)
    overdamped : bool
        If True, damping is in overdamped notation
        
    Returns
    -------
    tuple
        (amplitude, real_part, imag_part)
    """
    twopi = 2.0 * PI
    omega = twopi * freq
    omega0 = twopi / period
    
    if not overdamped:
        # Underdamped case: damping from log decrement
        a = np.log(damping)
        b = np.sqrt(PI * PI + a * a)
        eps = omega0 * a / b
    else:
        eps = omega0 * abs(damping)
    
    # Compute response using Aki & Richards formulation
    denom_real = omega * omega - omega0 * omega0
    denom_imag = -2.0 * eps * omega
    denom_sq = denom_real * denom_real + denom_imag * denom_imag
    
    # Response for velocity output
    num_real = -omega * omega * magnification
    
    if denom_sq > 0:
        # Complex division
        response_real = num_real * denom_real / denom_sq
        response_imag = -num_real * denom_imag / denom_sq
        
        # Flip sign for polarity
        response_real = -response_real
        response_imag = -response_imag
        
        r = np.sqrt(response_real ** 2 + response_imag ** 2)
        if r > 0:
            are = response_real / r
            aim = response_imag / r
        else:
            are, aim = 0.0, 0.0
    else:
        r, are, aim = 0.0, 0.0, 0.0
    
    return r, are, aim


def strainmeter_response(freq: float, magnification: float, damping: float,
                         period: float, phase_velocity: float = 5.0e5) -> Tuple[float, float]:
    """
    Response of strainmeter instrument.
    
    Equivalent to Fortran MYSTRN subroutine.
    
    Parameters
    ----------
    freq : float
        Frequency in Hz
    magnification : float
        Static magnification
    damping : float
        Damping constant
    period : float
        Natural period (seconds)
    phase_velocity : float
        Phase velocity (cm/s), default 5 km/s
        
    Returns
    -------
    tuple
        (amplitude, phase) in radians
    """
    omega = 2.0 * PI * freq
    omega0 = 2.0 * PI / period
    
    a = 2.0 * damping * omega * omega0
    b = omega0 * omega0 - omega * omega
    c = b * b
    d = a * a
    
    resp = phase_velocity * magnification * omega / np.sqrt(c + d)
    arg = 0.5 * PI - np.arctan2(a, b)
    
    return resp, arg


# =============================================================================
# TAHITI broadband instruments
# =============================================================================
def tahiti_broadband_response(freq: float, gain: float = 10.0) -> Tuple[float, float]:
    """
    Tahiti Very-Broadband (1-300s) instrument response.
    
    Equivalent to Fortran TAHBRB subroutine.
    
    Parameters
    ----------
    freq : float
        Frequency in Hz
    gain : float
        Instrument gain
        
    Returns
    -------
    tuple
        (amplitude, phase) in radians
    """
    sq2 = np.sqrt(2.0)
    omega = 2.0 * PI * freq
    zp = complex(0, omega)
    
    anumb = 40.0 * 634.0 * 335610.0
    
    za = zp * zp + complex(2 * PI * sq2, 0) * zp + complex(4 * PI * PI, 0)
    zc = complex(anumb, 0) * zp * zp * zp
    z1 = complex(1.0, 0)
    z634p = complex(634.0, 0) * zp
    z63p = complex(63.4, 0) * zp
    
    zd = (z1 + z634p) * (z1 + z63p) * (z1 + z63p)
    ze = complex(335610.0, 0) * zp * zp + complex(819.28, 0) * zp + z1
    
    zy = zc * complex(-omega * omega, 0) / (za * zd * ze)
    
    r = abs(zy) * gain / 2.51941e-4
    arg = np.angle(zy)
    
    return r, arg


def tahiti_digital_response(freq: float, t0: float = 1.0, tc: float = 60.0,
                            th: float = 4000.0, gain: float = 40000.0) -> Tuple[float, float]:
    """
    Tahiti digital instrument response.
    
    Equivalent to Fortran TAHDIG subroutine.
    
    Parameters
    ----------
    freq : float
        Frequency in Hz
    t0 : float
        Natural period (seconds)
    tc : float
        Corner period (seconds)
    th : float
        High-frequency period (seconds)
    gain : float
        Instrument gain
        
    Returns
    -------
    tuple
        (amplitude, phase) in radians
    """
    xi = 1.0 / np.sqrt(2.0)
    tref = 20.0
    
    omc = 2.0 * PI / tc
    om0 = 2.0 * PI / t0
    omh = 2.0 * PI / th
    
    # Calculate at reference frequency first
    omega_ref = 2.0 * PI / tref
    zp = complex(0, omega_ref)
    
    # Seismometer response
    zd = zp * zp + zp * complex(2.0 * xi * om0, 0) + complex(om0 * om0, 0)
    zs = zp * zp / zd
    
    # Low-pass filter
    zd = zp * zp + zp * complex(2.0 * xi * omc, 0) + complex(omc * omc, 0)
    zdi = complex(omc * omc, 0) / zd
    
    # High-pass filter
    zd = zp + complex(omh, 0)
    zh = zp / zd
    
    zresp_ref = zs * zdi * zh
    gref = abs(zresp_ref)
    
    # Calculate at requested frequency
    omega = 2.0 * PI * freq
    zp = complex(0, omega)
    
    zd = zp * zp + zp * complex(2.0 * xi * om0, 0) + complex(om0 * om0, 0)
    zs = zp * zp / zd
    
    zd = zp * zp + zp * complex(2.0 * xi * omc, 0) + complex(omc * omc, 0)
    zdi = complex(omc * omc, 0) / zd
    
    zd = zp + complex(omh, 0)
    zh = zp / zd
    
    zresp = zs * zdi * zh * complex(gain / gref, 0)
    
    r = abs(zresp)
    arg = np.angle(zresp)
    
    return r, arg


# =============================================================================
# PASSCAL and modern digital instruments
# =============================================================================
def passcal_response(freq: float, sensitivity: float = 1500.0,
                     corner_freq: float = 1.0/120.0) -> Tuple[float, float]:
    """
    PASSCAL instrument response.
    
    Parameters
    ----------
    freq : float
        Frequency in Hz
    sensitivity : float
        Sensitivity in V*s/m
    corner_freq : float
        Corner frequency in Hz
        
    Returns
    -------
    tuple
        (amplitude, phase) in radians
    """
    h = 1.0 / np.sqrt(2.0)
    f0 = corner_freq
    
    # Convert sensitivity to V*s/cm
    sens_cm = sensitivity / 100.0
    
    s_real = 1.0 - (f0 / freq) ** 2
    s_imag = -2.0 * h * f0 / freq
    s = complex(s_real, s_imag)
    
    u = complex(0, 2.0 * PI * freq * sens_cm) / s
    
    r = abs(u)
    arg = np.angle(u)
    
    return r, arg


def sro_iris_response(freq: float, station_type: str = 'PAS') -> Tuple[float, float]:
    """
    SRO/IRIS instrument response (placeholder - needs calibration files).
    
    Parameters
    ----------
    freq : float
        Frequency in Hz
    station_type : str
        Station type ('PAS' for Pasadena, 'HRV' for Harvard)
        
    Returns
    -------
    tuple
        (amplitude, phase)
    """
    # This would normally read from calibration files
    # Returning placeholder flat response
    return 1.0, 0.0


# =============================================================================
# File I/O for response files
# =============================================================================
def read_resp_file(filename: str) -> InstrumentResponse:
    """
    Read RESP format file (SEED/FDSN format).
    
    This is a simplified parser - full RESP files can be complex.
    
    Parameters
    ----------
    filename : str
        Path to RESP file
        
    Returns
    -------
    InstrumentResponse
        Instrument response parameters
    """
    a0 = 1.0
    sensitivity = 1.0
    zeros = []
    poles = []
    unit = 'M'
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('B053F03') or 'Transfer function type' in line:
            # Check for A (Laplace) or B (analog Hz)
            if 'A' in line.split(':')[-1]:
                response_type = 'A'
            else:
                response_type = 'B'
        
        elif line.startswith('B053F05') or 'Response in units' in line:
            parts = line.split(':')[-1].strip().upper()
            if 'M/S**2' in parts:
                unit = 'M/S**2'
            elif 'M/S' in parts:
                unit = 'M/S'
            else:
                unit = 'M'
        
        elif line.startswith('B053F07') or 'A0 normalization' in line:
            try:
                a0 = float(line.split(':')[-1].strip())
            except:
                pass
        
        elif line.startswith('B053F09') or 'Number of zeroes' in line:
            try:
                n_zeros = int(line.split(':')[-1].strip())
            except:
                n_zeros = 0
        
        elif line.startswith('B053F14') or 'Number of poles' in line:
            try:
                n_poles = int(line.split(':')[-1].strip())
            except:
                n_poles = 0
        
        elif line.startswith('B053F10') or 'Complex zeroes' in line:
            # Read zeros
            i += 1
            while i < len(lines) and len(zeros) < n_zeros:
                parts = lines[i].split()
                if len(parts) >= 3:
                    try:
                        real_part = float(parts[1])
                        imag_part = float(parts[2])
                        zeros.append(complex(real_part, imag_part))
                    except:
                        pass
                i += 1
            continue
        
        elif line.startswith('B053F15') or 'Complex poles' in line:
            # Read poles
            i += 1
            while i < len(lines) and len(poles) < n_poles:
                parts = lines[i].split()
                if len(parts) >= 3:
                    try:
                        real_part = float(parts[1])
                        imag_part = float(parts[2])
                        poles.append(complex(real_part, imag_part))
                    except:
                        pass
                i += 1
            continue
        
        elif line.startswith('B058F04') or 'Sensitivity' in line:
            try:
                sensitivity = float(line.split(':')[-1].strip())
            except:
                pass
        
        i += 1
    
    return InstrumentResponse(
        a0=a0,
        sensitivity=sensitivity,
        zeros=np.array(zeros),
        poles=np.array(poles),
        unit=unit
    )


def read_georom_format(filename: str) -> InstrumentResponse:
    """
    Read instrument response from EAO-Geoscope format file.
    
    This is the format used in the Fortran GeoRom subroutine.
    
    Parameters
    ----------
    filename : str
        Path to data file
        
    Returns
    -------
    InstrumentResponse
        Instrument response parameters
    """
    a0 = 1.0
    sensitivity = 1.0
    zeros_list = []
    poles_list = []
    unit = 'M'
    response_type = 'A'  # Laplace (rad/s)
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            header = parts[0][:8] if len(parts[0]) >= 8 else parts[0]
            
            if header == 'A0':
                a0 = float(parts[-1])
            
            elif header == 'AMPLIFAC':
                sensitivity = float(parts[-1])
            
            elif header.startswith('POLES'):
                # Format: POLES  A/B  start  end  values...
                ptype = parts[0][7] if len(parts[0]) > 7 else 'A'
                if ptype == 'B':
                    response_type = 'B'
                start = int(parts[1])
                end = int(parts[2])
                values = [float(x) for x in parts[3:]]
                for j in range(0, len(values) - 1, 2):
                    poles_list.append(complex(values[j], values[j + 1]))
            
            elif header.startswith('ZEROS'):
                ptype = parts[0][7] if len(parts[0]) > 7 else 'A'
                if ptype == 'B':
                    response_type = 'B'
                start = int(parts[1])
                end = int(parts[2])
                values = [float(x) for x in parts[3:]]
                for j in range(0, len(values) - 1, 2):
                    zeros_list.append(complex(values[j], values[j + 1]))
            
            elif header == 'UNIT':
                unit_str = ''.join(parts[1:]).upper()
                if 'M/S**2' in unit_str:
                    unit = 'M/S**2'
                elif 'M/S' in unit_str:
                    unit = 'M/S'
                else:
                    unit = 'M'
            
            elif header == 'ENDOFILE':
                break
    
    zeros = np.array(zeros_list)
    poles = np.array(poles_list)
    
    # Convert from Hz to rad/s if format is B
    if response_type == 'B':
        zeros = zeros * (2.0 * PI)
        poles = poles * (2.0 * PI)
        n_poles = len(poles)
        n_zeros = len(zeros)
        a0 = a0 * ((2.0 * PI) ** (n_poles - n_zeros))
    
    return InstrumentResponse(
        a0=a0,
        sensitivity=sensitivity,
        zeros=zeros,
        poles=poles,
        unit=unit
    )


def add_integration_zeros(response: InstrumentResponse, n_integrations: int = 1) -> InstrumentResponse:
    """
    Add zeros at origin for integration (velocity to displacement conversion).
    
    Parameters
    ----------
    response : InstrumentResponse
        Original instrument response
    n_integrations : int
        Number of integrations (1 for velocity to displacement, 2 for acceleration)
        
    Returns
    -------
    InstrumentResponse
        Modified response with added zeros
    """
    new_zeros = np.zeros(len(response.zeros) + n_integrations, dtype=complex)
    new_zeros[:len(response.zeros)] = response.zeros
    # New zeros are at the origin (0 + 0j)
    
    return InstrumentResponse(
        a0=response.a0,
        sensitivity=response.sensitivity,
        zeros=new_zeros,
        poles=response.poles,
        unit=response.unit
    )


# =============================================================================
# Standard instrument configurations (from MYZGIL)
# =============================================================================
STANDARD_INSTRUMENTS = {
    'WWSSN_LP_15_100': {'tp': 15.0, 'tg': 100.0, 'coupling': 0.5, 'mag': 1500.0},
    'WWSSN_LP_30_100': {'tp': 30.0, 'tg': 100.0, 'coupling': 0.5, 'mag': 1500.0},
    'WWSSN_SP': {'tp': 1.0, 'tg': 0.8, 'coupling': 0.05, 'mag': 50000.0},
    'PAS_1_90': {'tp': 1.0, 'tg': 90.0, 'coupling': 0.05, 'mag': 3000.0},
    'PRESS_EWING_30_90': {'tp': 30.0, 'tg': 90.0, 'coupling': 0.0, 'mag': 2200.0},
    'GALITZIN_12_12': {'tp': 12.0, 'tg': 12.0, 'coupling': 0.0, 'mag': 740.0},
    'WOOD_ANDERSON': {'period': 0.8, 'mag': 2800.0, 'damping': 0.8},
}


def get_standard_instrument_response(name: str, freq: float) -> Tuple[float, float, float]:
    """
    Get response for a standard instrument configuration.
    
    Parameters
    ----------
    name : str
        Instrument name (from STANDARD_INSTRUMENTS)
    freq : float
        Frequency in Hz
        
    Returns
    -------
    tuple
        (amplitude, real_part, imag_part)
    """
    if name not in STANDARD_INSTRUMENTS:
        raise ValueError(f"Unknown instrument: {name}")
    
    config = STANDARD_INSTRUMENTS[name]
    
    if 'tp' in config:
        # WWSSN-type electromagnetic
        return wwssn_lp_response(freq, config['tp'], config['tg'], 
                                 config.get('coupling', 0.0))
    else:
        # Mechanical seismograph
        return mechanical_seismograph_response(freq, config['period'],
                                               config['mag'], config['damping'])


if __name__ == "__main__":
    print("Testing instrument_response.py")
    
    # Test WWSSN response
    freq = 0.05  # 20 second period
    r, are, aim = wwssn_lp_response(freq, tp=15.0, tg=100.0, coupling=0.5)
    print(f"WWSSN LP at {freq} Hz: amplitude={r:.4f}, phase={np.arctan2(aim, are)*180/PI:.2f}°")
    
    # Test poles/zeros response
    zeros = np.array([0+0j, 0+0j])
    poles = np.array([-0.01234+0.01234j, -0.01234-0.01234j, -39.18+49.12j, -39.18-49.12j])
    gain = 1.0e9
    
    resp = compute_response(0.1, zeros, poles, gain)
    print(f"Poles/zeros response at 0.1 Hz: |H|={abs(resp):.2e}, phase={np.angle(resp)*180/PI:.2f}°")
    
    print("All tests passed!")

