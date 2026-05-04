#!/usr/bin/env python3
"""Validate the 2014 New Zealand event at IU.TUC against a published theta."""

import os
import sys

import numpy as np
from obspy import UTCDateTime, read, read_inventory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from energy_calculation import compute_seismic_energy
from instrument_response import InstrumentResponse
from travel_time import JBTables


EVENT = {
    "origin": UTCDateTime("2014-11-16T22:33:20.5"),
    "centroid": UTCDateTime("2014-11-16T22:33:28.9"),
    "lat": -37.51,
    "lon": 179.74,
    "depth_km": 20.9,
    "moment": 1.32e26,
    "strike": 42.0,
    "dip": 34.0,
    "rake": -82.0,
}

STATION = {
    "network": "IU",
    "station": "TUC",
    "location": "00",
    "channel": "BHZ",
    "lat": 32.3098,
    "lon": -110.7847,
}

MSEED = "2014-11-16-mw67-off-e-coast-of-n-island-nz.miniseed"
STATIONXML = "IU.TUC.00.BHZ.xml"
PUBLISHED_THETA = -4.49


def corr_nz(distance_deg: float) -> float:
    """New Zealand high-distance correction from Okal & Saloor (2017), eq. 5a."""
    return 0.395 + 0.147 * (distance_deg - 90.0)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    mseed_path = os.path.join(here, MSEED)
    xml_path = os.path.join(here, STATIONXML)

    jb = JBTables()
    from seismic_utils import great_circle

    dist, az, _, _ = great_circle(
        EVENT["lat"], EVENT["lon"], STATION["lat"], STATION["lon"]
    )
    travel_time = jb.get_travel_time(dist, EVENT["depth_km"])
    p_arrival = EVENT["origin"] + travel_time

    st = read(mseed_path)
    st = st.select(
        network=STATION["network"],
        station=STATION["station"],
        location=STATION["location"],
        channel=STATION["channel"],
    )
    if len(st) != 1:
        raise RuntimeError(f"Expected one IU.TUC.00.BHZ trace, found {len(st)}")

    inv = read_inventory(xml_path)

    # Fortran endig.f uses P - 10 s to P + 60 s for this event size.
    window_start = p_arrival - 10.0
    window_end = p_arrival + 60.0
    st.trim(window_start, window_end, pad=False)
    st.detrend("demean")
    st.detrend("linear")

    # Convert counts to ground velocity. The downstream energy routine then uses
    # a flat M/S response, matching the already-corrected units.
    st_vel = st.copy()
    st_vel.remove_response(
        inventory=inv,
        output="VEL",
        pre_filt=(0.05, 0.08, 2.5, 4.0),
        water_level=60,
    )

    tr = st_vel[0]
    data = tr.data.astype(float)
    instrument = InstrumentResponse(
        a0=1.0,
        sensitivity=1.0,
        zeros=np.array([], dtype=complex),
        poles=np.array([], dtype=complex),
        unit="M/S",
    )

    result = compute_seismic_energy(
        data=data,
        dt=tr.stats.delta,
        instrument=instrument,
        elat=EVENT["lat"],
        elon=EVENT["lon"],
        slat=STATION["lat"],
        slon=STATION["lon"],
        depth_km=EVENT["depth_km"],
        moment=EVENT["moment"],
        strike=EVENT["strike"],
        dip=EVENT["dip"],
        rake=EVENT["rake"],
        tmin=0.5,
        tmax=10.0,
        station_code="IU.TUC.00.BHZ",
        jb_tables=jb,
    )

    correction = corr_nz(result.distance_deg)
    theta_corrected = result.theta_estimated + correction

    print("\nVALIDATION TARGET")
    print(f"Published IU.TUC 2014 theta: {PUBLISHED_THETA:.2f}")
    print(f"Python raw theta:             {result.theta_estimated:.2f}")
    print(f"CorrNZ:                       {correction:.2f}")
    print(f"Python corrected theta:       {theta_corrected:.2f}")
    print(f"Corrected residual:           {theta_corrected - PUBLISHED_THETA:+.2f}")
    print(f"Distance:                     {result.distance_deg:.2f} deg")
    print(f"P arrival used:               {p_arrival.isoformat()}")
    print(f"Window:                       {window_start.isoformat()} to {window_end.isoformat()}")


if __name__ == "__main__":
    main()
