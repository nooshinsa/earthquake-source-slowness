# THETA Calculator - Python Implementation

Python conversion of Fortran seismological routines for computing **slowness parameters** and **seismic radiated energy**.

## Overview

This package computes:
- **Slowness (ray parameter)** `p` - the rate of change of travel time with distance
- **Seismic radiated energy** `E` - energy released during an earthquake
- **Theta parameter** `Θ = log10(E/M0)` - energy-to-moment ratio

Based on the methodology of:
- Boatwright & Choy (1986) - Energy calculation
- Jeffreys-Bullen tables - Travel times
- Aki & Richards - Wave propagation theory

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Demo with synthetic data
```bash
python theta_calculator.py --demo
```

### Process real data
```bash
python theta_calculator.py \
    --data seismogram.dat \
    --response response.dat \
    --epicenter epicenter.dat \
    --moment 1e27 \
    --strike 0 --dip 45 --rake 90
```

## Module Structure

### `seismic_utils.py`
Core mathematical and geophysical utilities:
- FFT operations
- Great circle distance calculations
- Geodesic computations
- Signal processing (detrend, taper, filter)
- Statistical functions
- Time utilities

### `instrument_response.py`
Seismometer instrument response handling:
- Poles and zeros representation
- Frequency response calculation
- WWSSN, mechanical, and digital instrument models
- RESP file reading

### `travel_time.py`
P-wave travel times and slowness:
- Jeffreys-Bullen table interpolation
- Ray parameter computation
- Geometric spreading
- Radiation pattern coefficients
- t* attenuation

### `energy_calculation.py`
Main energy computation module:
- Seismic radiated energy calculation
- Theta parameter computation
- Event classification (slow/normal/fast)

### `theta_calculator.py`
Command-line interface and demo script.

## Usage Examples

### Computing Slowness

```python
from travel_time import JBTables, slowness_to_different_units

# Initialize JB tables
jb = JBTables()

# Get slowness at 60° distance, 33 km depth
distance = 60.0  # degrees
depth = 33.0     # km

slowness = jb.get_slowness(distance, depth)
print(f"Slowness: {slowness:.4f} s/deg")

# Convert to other units
p = slowness_to_different_units(slowness)
print(f"p = {p['p_km']:.6f} s/km")
print(f"p = {p['p_rad']:.2f} s/rad")
```

### Computing Great Circle Path

```python
from seismic_utils import great_circle

# Epicenter and station coordinates
elat, elon = -15.0, -75.0   # Peru
slat, slon = 34.0, -118.0   # Pasadena

dist, az12, az21, gc = great_circle(elat, elon, slat, slon)
print(f"Distance: {dist:.2f}°")
print(f"Azimuth: {az12:.2f}°")
```

### Computing Radiation Pattern

```python
from travel_time import compute_radiation_coefficients

coeffs = compute_radiation_coefficients(
    distance_deg=60.0,
    depth_km=33.0,
    azimuth=45.0,
    strike=0.0,
    dip=45.0,
    rake=90.0  # Thrust fault
)

print(f"Direct P radiation: {coeffs['Fp']:.3f}")
print(f"Geometric spreading: {coeffs['g']:.4f}")
```

### Full Energy Calculation

```python
import numpy as np
from instrument_response import InstrumentResponse
from energy_calculation import compute_seismic_energy, classify_event

# Create instrument response
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

# Your seismogram data
data = np.loadtxt('seismogram.txt')
dt = 0.5  # sampling interval

# Compute energy
result = compute_seismic_energy(
    data=data,
    dt=dt,
    instrument=instrument,
    elat=-15.0, elon=-75.0,  # epicenter
    slat=34.0, slon=-118.0,   # station
    depth_km=33.0,
    moment=1.0e27,  # dyn-cm
    strike=0.0, dip=45.0, rake=90.0
)

print(f"Theta: {result.theta_estimated:.2f}")
print(f"Classification: {classify_event(result.theta_estimated)}")
```

## Physical Background

### Slowness (Ray Parameter)
The slowness `p = dT/dΔ` represents how travel time changes with epicentral distance. It's related to the takeoff angle by Snell's law:

```
p = sin(i) / v
```

where `i` is the takeoff angle and `v` is the seismic velocity.

### Theta Parameter
The theta parameter measures earthquake "speed":

```
Θ = log10(E/M0)
```

- **Θ < -5.75**: Slow earthquake (tsunami earthquake potential)
- **-5.75 < Θ < -4.30**: Normal earthquake
- **Θ > -4.30**: Fast earthquake (high stress drop)

### Energy Calculation
Radiated energy is computed from velocity spectra using equation (17) of Boatwright & Choy (1986):

```
E = (4π ρ v / F²gP²) × ∫|v(ω)|² e^(ω t*) dω
```

Where:
- ρ = density
- v = P-wave velocity
- F²gP² = radiation pattern factor
- v(ω) = velocity spectrum
- t* = attenuation parameter

## Fortran to Python Mapping

| Fortran Routine | Python Function/Class |
|----------------|----------------------|
| COOLB | `np.fft.fft()` |
| MYGRT | `great_circle()` |
| MYFND | `find_fft_size()` |
| MYDTR | `detrend()` |
| MYTPR | `cosine_taper()` |
| MYSERT | `interpolate_in_array()` |
| MYWWSS | `wwssn_lp_response()` |
| MYRSMS | `mechanical_seismograph_response()` |
| BC10 | `compute_radiation_coefficients()` |
| GEORPZ | `compute_geometric_spreading()` |
| GeoRom | `read_georom_format()` |
| instrument | `compute_response()` |
| endig | `compute_seismic_energy()` |

## Testing

Run the demo to verify installation:

```bash
python theta_calculator.py --demo
```

Run module tests:

```bash
python seismic_utils.py
python instrument_response.py
python travel_time.py
python energy_calculation.py
```

## References

1. Boatwright, J. and G. L. Choy (1986). Teleseismic estimates of the energy radiated by shallow earthquakes. *J. Geophys. Res.*, 91, 2095-2112.

2. Kanamori, H. and G. S. Stewart (1976). Mode of the strain release along the Gibbs Fracture zone, Mid-Atlantic Ridge. *Phys. Earth Planet. Inter.*, 11, 312-332.

3. Choy, G. L. and V. F. Cormier (1986). Direct measurement of the mantle attenuation operator from broadband P and S waveforms. *J. Geophys. Res.*, 91, 7326-7342.

4. Aki, K. and P. G. Richards (2002). *Quantitative Seismology*, 2nd ed. University Science Books.

