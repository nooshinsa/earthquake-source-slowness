# Earthquake Source Slowness (THETA) Python

Python implementation for calculating earthquake source slowness, also referred
to here as `Theta`, through the THETA seismic radiated-energy workflow
originally used in the Fortran code.

The code computes station-by-station:

- P-wave ray parameter `p`
- radiated seismic energy `E`
- earthquake source slowness parameter `Theta = log10(E / M0)`

The most validated workflow currently reads the original THETA/GeoRom-style
event folders and writes a station-by-station CSV table.

## What You Need

For a calculation run you need:

- an event folder in the format described below
- scalar moment `M0`
- focal mechanism: strike, dip, and rake
- station waveform files with embedded GeoRom response information

You do not need the original Fortran code to run the Python version.

## Installation

Create or activate a Python environment, then install dependencies:

```bash
pip install -r python_code/requirements.txt
```

ObsPy is only required for downloading MiniSEED/StationXML data. The
Fortran-style folder workflow mainly depends on NumPy and SciPy.

## Input Folder Format

For a Fortran-style event folder, prepare:

```text
EVENT_FOLDER/
  Epicenter.parameters
  records
  station_file_1
  station_file_2
  ...
  results.en              optional Fortran comparison table
```

`Epicenter.parameters` contains event latitude, longitude, depth, origin time,
and a text description:

```text
-17.74 -65.90 381
05 03 50.1
15 MAR 2019 -- 05:03 -- BOLIVIA
19074
```

`records` lists the station files to process:

```text
19074.WVTZ
19074.TUCZ
19074.CMLZ
```

Each station file should contain the waveform samples and embedded GeoRom
instrument response information, including station coordinates, poles, zeros,
gain, unit, and `ENDOFILE`.

## Run an Event Folder

From the repository root:

```bash
python3 python_code/theta_calculator.py run-folder EVENT_FOLDER \
  --moment 4.2e25 \
  --strike 55 \
  --dip 15 \
  --rake 92
```

By default, the output is written to:

```text
EVENT_FOLDER/theta_python_results.csv
```

The CSV includes station code, distance, azimuth, ray parameter `p`, estimated
Theta, true-mechanism Theta, energy values, and optional Python-minus-Fortran
residuals if `results.en` is present.

## About Large Strike Values

Some original Fortran folders use large values such as:

```text
strike = 2215
dip = 15
rake = 92
```

This large strike value comes from the historical Fortran workflow, where the
angle is used inside sine and cosine calculations. Since trigonometric
functions are periodic, `2215 degrees` is equivalent to:

```text
2215 mod 360 = 55 degrees
```

For new runs, it is clearer to use the normalized value:

```text
--strike 55 --dip 15 --rake 92
```

The old value should still produce equivalent trigonometric behavior, but the
normalized angle is easier to understand and publish.

## Download Data

The code can download MiniSEED waveforms and StationXML responses using ObsPy:

```bash
python3 python_code/theta_calculator.py download-event \
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
  --output-dir python_code/theta_results
```

Download filters currently include:

- network codes, for example `II,IU`
- channel, for example `BHZ`
- minimum and maximum epicentral distance
- time window around the origin
- FDSN client, default `IRIS`

The download command collects data. The most validated calculation command is
still `run-folder`.

## Process Downloaded Data

Downloaded MiniSEED/StationXML folders can be processed with:

```bash
python3 python_code/theta_calculator.py process-downloaded \
  downloads_test/BOL_2019 \
  --min-distance 35 \
  --max-distance 80 \
  --remove-outliers
```

This writes:

```text
downloads_test/BOL_2019/theta_downloaded_results.csv
```

The `--remove-outliers` option marks robust Theta outliers in the CSV and
excludes them from the reported event mean. The station rows are still kept in
the output table for inspection. This downloaded-data calculation path is newer
than the Fortran-style `run-folder` workflow and should be validated carefully.

## Bulk Catalog Workflow

For many events, first download and combine the Global CMT NDK catalog files:

```bash
python3 python_code/theta_calculator.py download-gcmt-catalog \
  --output catalogs/globalcmt_1976_2025sep.ndk \
  --end-year 2025 \
  --end-month 9 \
  --allow-missing
```

Then convert the NDK file to the CSV format used by this package:

```bash
python3 python_code/theta_calculator.py make-catalog catalogs/globalcmt_1976_2025sep.ndk \
  --output catalogs/theta_catalog.csv \
  --start-date 2000-01-01 \
  --end-date 2025-09-30 \
  --min-magnitude 6.0 \
  --min-depth 0 \
  --max-depth 700
```

The generated CSV contains:

```text
event_id,origin_time,latitude,longitude,depth_km,magnitude,moment,strike,dip,rake
```

Then download waveforms/responses and calculate Theta for every event:

```bash
python3 python_code/theta_calculator.py process-catalog catalogs/theta_catalog.csv \
  --output-dir bulk_results \
  --networks II,IU \
  --channel BHZ \
  --min-distance 35 \
  --max-distance 80 \
  --remove-outliers
```

This creates one folder per event and writes a master summary:

```text
bulk_results/
  EVENT_ID/
    event_info.txt
    *.mseed
    *.xml
    *.meta
    theta_downloaded_results.csv
  theta_summary.csv
  theta_all_stations.csv
```

`theta_summary.csv` has one row per event. `theta_all_stations.csv` has every
station Theta value for every processed event, with the event information
included on each row. Theta columns are written with two decimal places in the
CSV output files.

If the event folders are already downloaded, add `--no-download` to reprocess
them without requesting data again.

## Validation Status

The Python calculation has been compared with a local Fortran BOL_19 result set.
For the stations present in the Fortran `results.en` file, the comparison gave:

```text
Mean Python-minus-Fortran estimated-Theta residual: about -0.01
Median absolute residual: about 0.04
```

That validation dataset is not included in the public repository because it
contains large waveform and response files. It can be added later as a separate
example or comparison dataset.

## Repository Contents

```text
python_code/theta_calculator.py       command-line interface
python_code/energy_calculation.py     core energy and Theta calculation
python_code/travel_time.py            travel times, ray parameter p, radiation terms
python_code/geometric_spreading.py    Fortran-style geometric spreading
python_code/instrument_response.py    poles/zeros and GeoRom response parsing
python_code/seismic_utils.py          geometry, FFT, tapering, utility functions
python_code/depth_bins.py             depth-dependent correction parameters
python_code/iris_downloader.py        MiniSEED/StationXML download helpers
python_code/requirements.txt          Python dependencies
```
