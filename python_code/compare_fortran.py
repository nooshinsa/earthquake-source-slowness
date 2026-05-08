#!/usr/bin/env python3
"""
DETAILED COMPARISON: Python vs Fortran

This script prints ALL intermediate values so we can find
where Python and Fortran calculations diverge.
"""

import numpy as np

print("="*70)
print("DETAILED CALCULATION COMPARISON")
print("="*70)

# ============================================================
# EVENT PARAMETERS (from CMT)
# ============================================================
print("\n1. EVENT PARAMETERS")
print("-"*70)

event_lat = -11.18
event_lon = 165.21
depth_km = 20.2
M0 = 9.37e27  # dyn-cm
Mw = 7.9
strike = 320
dip = 20
rake = 89

print(f"   Latitude:  {event_lat}°")
print(f"   Longitude: {event_lon}°")
print(f"   Depth:     {depth_km} km")
print(f"   M0:        {M0:.2e} dyn-cm")
print(f"   Mw:        {Mw}")
print(f"   Strike:    {strike}°")
print(f"   Dip:       {dip}°")
print(f"   Rake:      {rake}°")

# ============================================================
# STATION PARAMETERS (II.COCO from debug output)
# ============================================================
print("\n2. STATION PARAMETERS (II.COCO)")
print("-"*70)

sta_lat = -12.19  # approximate
sta_lon = 96.83   # approximate (Cocos Islands)
distance_deg = 66.78
azimuth = 261.41

print(f"   Distance:  {distance_deg}°")
print(f"   Azimuth:   {azimuth}°")

# ============================================================
# DEPTH BIN PARAMETERS
# ============================================================
print("\n3. DEPTH BIN (SHALLOW: 0-80 km)")
print("-"*70)

# From depth_bins.py
a0 = 1.171
a1 = -7.271e-3
a2 = 6.009e-5
gamma_h = 1.0  # depth correction factor
t_star = 1.0   # attenuation
window_duration = 70.0  # seconds

print(f"   Coefficients: a0={a0}, a1={a1}, a2={a2}")
print(f"   gamma_h:      {gamma_h}")
print(f"   t*:           {t_star} s")
print(f"   Window:       {window_duration} s")

# ============================================================
# FgP² CALCULATION
# ============================================================
print("\n4. RADIATION PATTERN FgP²")
print("-"*70)

# Distance-based estimation (from Saloor & Okal 2018)
delta = distance_deg
FgP2_estimated = a0 + a1 * delta + a2 * delta**2

# With depth correction
FgP2_corrected = FgP2_estimated * gamma_h

print(f"   FgP²(Δ) = a0 + a1*Δ + a2*Δ²")
print(f"           = {a0} + {a1}*{delta} + {a2}*{delta}²")
print(f"           = {FgP2_estimated:.4f}")
print(f"   FgP² (depth corrected) = {FgP2_corrected:.4f}")
print(f"")
print(f"   >>> FORTRAN FgP²: ???")
print(f"   >>> Does your Fortran use the same formula?")

# ============================================================
# SLOWNESS / RAY PARAMETER
# ============================================================
print("\n5. SLOWNESS (Ray Parameter)")
print("-"*70)

from travel_time import JBTables
jb = JBTables()
slowness_deg = jb.get_slowness(distance_deg, depth_km)
slowness_km = slowness_deg / 111.195  # deg to km conversion
slowness_rad = slowness_deg * 180 / np.pi  # to s/rad

print(f"   p = {slowness_deg:.4f} s/deg")
print(f"   p = {slowness_km:.6f} s/km")
print(f"   p = {slowness_rad:.2f} s/rad")
print(f"")
print(f"   >>> FORTRAN ray parameter p: ???")

# ============================================================
# GEOMETRIC SPREADING
# ============================================================
print("\n6. GEOMETRIC SPREADING")
print("-"*70)

from geometric_spreading import GeometricSpreadingTables, compute_geometric_spreading_fortran
d2t_tables = GeometricSpreadingTables()
g_spread, rpz = compute_geometric_spreading_fortran(
    distance_deg, depth_km, slowness_deg, d2t_tables=d2t_tables
)

print(f"   g(Δ, h) = {g_spread:.4e}")
print(f"")
print(f"   >>> FORTRAN geometric spreading: ???")

# ============================================================
# RECEIVER FUNCTION
# ============================================================
print("\n7. RECEIVER FUNCTION (rpz)")
print("-"*70)

Vp_surface = 5.8  # km/s

print(f"   Vp (surface) = {Vp_surface} km/s")
print(f"   rpz = {rpz:.4f}")
print(f"")
print(f"   >>> FORTRAN rpz: ???")

# ============================================================
# THETA FORMULA
# ============================================================
print("\n8. THETA FORMULA")
print("-"*70)
print("""
   Θ = log10(E / M0)
   
   where E is integrated from:
   
   E = 4π ρ c³ r² * ∫ v(t)² dt * corrections
   
   Corrections include:
   - Geometric spreading: g
   - Receiver function: rpz  
   - Radiation pattern: FgP²
   - Attenuation: exp(πf*t*)
   - Reflection coefficients: PP, SP
""")

# ============================================================
# POSSIBLE SOURCES OF DIFFERENCE
# ============================================================
print("\n9. POSSIBLE SOURCES OF 2.0 DIFFERENCE")
print("-"*70)
print("""
   A 2.0 difference in Θ means 100× difference in E/M0.
   
   Possible causes:
   
   1. FgP² ESTIMATION
      - Python uses: Saloor & Okal (2018) polynomial
      - Fortran uses: ???
      
   2. GEOMETRIC SPREADING
      - Different g(Δ,h) formula?
      
   3. ATTENUATION (t*)
      - Python t* = 1.0 s (shallow)
      - Fortran t* = ???
      
   4. FREQUENCY BAND
      - Python: 0.1 - 2.0 Hz (0.5-10 s)
      - Fortran: ???
      
   5. TIME WINDOW
      - Python: 70 s
      - Fortran: ???
      
   6. UNIT CONVERSION
      - Energy: ergs vs joules?
      - M0: dyn-cm vs N-m?
""")

# ============================================================
# QUESTIONS FOR YOU
# ============================================================
print("\n" + "="*70)
print("QUESTIONS TO COMPARE WITH YOUR FORTRAN:")
print("="*70)
print("""
Please check your Fortran code for these values:

1. What is FgP² for this station (II.COCO, Δ=66.78°)?
   Python: {:.4f}

2. What geometric spreading formula do you use?
   Python g: {:.4e}

3. What is t* (attenuation) for shallow events?
   Python: 1.0 s

4. What frequency band?
   Python: 0.1-2.0 Hz (periods 0.5-10 s)

5. What time window after P arrival?
   Python: 70 s

6. How do you estimate FgP²? 
   - True focal mechanism?
   - Distance-based polynomial?
   - Fixed value?
""".format(FgP2_corrected, g_spread))

print("="*70)

