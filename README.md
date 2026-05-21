# Earthquake Source Slowness (THETA) Python

Python tools for downloading seismic waveform data and calculating earthquake
source slowness, here reported as:

```text
Theta = log10(E / M0)
```

The code computes station-by-station radiated seismic energy, P-wave ray
parameter `p`, and event-level Theta summaries. The main workflow uses
MiniSEED waveforms, StationXML instrument responses, and Global CMT event
metadata.

This repository is a Python implementation of a THETA radiated-energy workflow
originally used in Fortran. The Fortran comparison is used for validation, but
users do not need the Fortran code to run the Python workflow.

## Installation

Create or activate a Python environment, then install dependencies:

```bash
pip install -r python_code/requirements.txt
```

ObsPy is required for downloading waveforms, reading Global CMT/NDK catalogs,
and handling MiniSEED/StationXML data.

## Main Commands

All actions use the same entry point:

```bash
python python_code/theta_calculator.py COMMAND [OPTIONS]
```

Common commands are:

```text
download-gcmt-catalog   download and combine Global CMT NDK files
make-catalog            convert Global CMT/ObsPy catalog to THETA CSV
process-catalog         bulk download data and calculate Theta
download-event          download one event
process-downloaded      calculate Theta for one downloaded event folder
```

## Bulk Workflow

For many events, first download and combine Global CMT NDK files:

```bash
python python_code/theta_calculator.py download-gcmt-catalog \
  --output catalogs/globalcmt_1976_2025sep.ndk \
  --end-year 2025 \
  --end-month 9 \
  --allow-missing
```

Then convert the NDK file to a THETA catalog CSV:

```bash
python python_code/theta_calculator.py make-catalog catalogs/globalcmt_1976_2025sep.ndk \
  --output catalogs/theta_catalog_2015_2025.csv \
  --start-date 2015-01-01 \
  --end-date 2025-09-30 \
  --min-magnitude 6.0 \
  --min-depth 0 \
  --max-depth 700
```

The generated CSV contains:

```text
event_id,origin_time,latitude,longitude,depth_km,magnitude,moment,strike,dip,rake
```

Run the bulk calculation:

```bash
python python_code/theta_calculator.py process-catalog catalogs/theta_catalog_2015_2025.csv \
  --output-dir bulk_results_2015_2025 \
  --networks II,IU \
  --channel BHZ \
  --min-distance 35 \
  --max-distance 80 \
  --remove-outliers
```

For a short test run, add `--limit`:

```bash
python python_code/theta_calculator.py process-catalog catalogs/theta_catalog_2015_2025.csv \
  --output-dir bulk_results_test \
  --remove-outliers \
  --limit 5
```

If event folders are already downloaded and you only want to recalculate Theta,
add `--no-download`:

```bash
python python_code/theta_calculator.py process-catalog catalogs/theta_catalog_2015_2025.csv \
  --output-dir bulk_results_test \
  --remove-outliers \
  --limit 5 \
  --no-download
```

## Bulk Output

`process-catalog` creates one folder per event:

```text
bulk_results_2015_2025/
  EVENT_ID/
    event_info.txt
    *.mseed
    *.xml
    *.meta
    theta_downloaded_results.csv
  theta_summary.csv
  theta_all_stations.csv
```

`theta_summary.csv` has one row per event:

```text
event_id,origin_time,latitude,longitude,depth_km,magnitude,moment,
strike,dip,rake,n_stations,n_used,n_outliers,theta_mean,theta_std,
classification,status,error
```

`theta_all_stations.csv` has every station Theta value for every processed
event, with event information repeated on each row:

```text
event_id,origin_time,latitude,longitude,depth_km,magnitude,moment,
strike,dip,rake,station,distance_deg,azimuth_deg,ray_parameter_s_deg,
theta_estimated,theta_true_mech,energy_estimated_ergs,outlier,used_in_mean
```

Theta values in output CSV files are written with two decimal places.

## One Event Workflow

To download one event:

```bash
python python_code/theta_calculator.py download-event \
  --event-id BOL_2019 \
  --origin-time 2019-03-15T05:03:50.1 \
  --latitude -17.74 \
  --longitude -65.90 \
  --depth 381 \
  --magnitude 6.3 \
  --moment 4.2e25 \
  --strike 55 \
  --dip 15 \
  --rake 92 \
  --networks II,IU \
  --channel BHZ \
  --min-distance 35 \
  --max-distance 80 \
  --pre-origin 60 \
  --post-origin 1800 \
  --output-dir downloads_test
```

This creates:

```text
downloads_test/BOL_2019/
  event_info.txt
  *.mseed
  *.xml
  *.meta
```

Then calculate Theta:

```bash
python python_code/theta_calculator.py process-downloaded \
  downloads_test/BOL_2019 \
  --remove-outliers
```

This writes:

```text
downloads_test/BOL_2019/theta_downloaded_results.csv
```

The `event_info.txt` file stores event origin time, location, depth, moment,
and focal mechanism. The `.mseed` files store waveform data, the `.xml` files
store StationXML instrument responses, and the `.meta` files store station
metadata such as distance and azimuth.

## Output Columns

`theta_estimated` is the preferred station Theta value for event averaging. It
uses the estimated/statistical radiation correction.

`theta_true_mech` uses the supplied strike, dip, and rake for the station
radiation correction. Both values are written so users can compare them.

`outlier` marks robust station-level Theta outliers when `--remove-outliers` is
used. Outlier rows remain in the CSV, but `used_in_mean` is set to `False`.

## Validation

The Python workflow has been checked against a local Fortran BOL_19 validation
case. For matched stations, the Python-minus-Fortran estimated-Theta comparison
gave approximately:

```text
Mean residual: -0.01
Median absolute residual: 0.04
```

That validation dataset is not included in this public repository because it
contains waveform and response data.

## Legacy Folder Support

The package can still process older THETA/Fortran-style event folders with:

```bash
python python_code/theta_calculator.py run-folder EVENT_FOLDER \
  --moment 4.2e25 \
  --strike 55 \
  --dip 15 \
  --rake 92
```

This is mainly for validation and comparison with older datasets. New users
should start with the MiniSEED/StationXML and catalog workflows above.

## Repository Contents

```text
python_code/theta_calculator.py       command-line interface
python_code/energy_calculation.py     core energy and Theta calculation
python_code/travel_time.py            travel times, ray parameter p, radiation terms
python_code/geometric_spreading.py    geometric spreading correction
python_code/instrument_response.py    response and GeoRom parsing helpers
python_code/seismic_utils.py          geometry, FFT, tapering, utility functions
python_code/depth_bins.py             depth-dependent correction parameters
python_code/iris_downloader.py        MiniSEED/StationXML download helpers
python_code/requirements.txt          Python dependencies
```
