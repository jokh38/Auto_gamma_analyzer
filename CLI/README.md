# Auto Gamma Analysis CLI Tool

A command-line tool for performing 2D Gamma Analysis on DICOM RT dose files and MCC measurement data.

## Prerequisites

- Python 3.6+
- numpy
- matplotlib
- pydicom
- pyyaml
- pylinac
- scipy

You can install the dependencies using pip:
```bash
pip install -r requirements.txt
```

## File Structure

- `src/main.py`: Main command-line application entry point.
- `src/file_handlers.py`: Classes for handling DICOM and MCC files.
- `src/analysis.py`: Profile extraction and gamma analysis functions.
- `src/reporting.py`: Functions for generating image reports.
- `src/utils.py`: Utility functions and logging setup.

## Configuration

The analysis parameters are configured through a `config.yaml` file in the root directory.

```yaml
dta: 3
dd: 3
suppression_level: 10
save_csv: true
csv_export_path: "csv_exports"
```

- `dta`: Distance-to-agreement in millimeters (e.g., 3).
- `dd`: Dose-difference in percentage (e.g., 3).
- `suppression_level`: The dose level (as a percentage of the maximum dose) below which data is not considered in the analysis (e.g., 10).


## Usage

To run the analysis, use the `src/main.py` script with the following arguments:

- `--rtplan`: Path to the DICOM RT plan file.
- `--measure`: Path to the measurement data file (MCC).

### Example

The repository includes sample data in the `data/` directory.

```bash
python src/main.py --rtplan data/1G240_2cm.dcm --measure data/1G240_2cm.mcc
python src/main.py --rtplan data/104659/RD.1G.dcm --measure data/104659/1G_020mm.mcc
```

The tool will print the gamma analysis results to the console and save a report image in the `reports/` directory.

## Library Reference

- numpy (1.21.0): Used for array operations and calculations.
- matplotlib (3.5.1): Used for visualization and plotting.
- pydicom (2.3.0): Used for loading and handling DICOM files.
- scipy (1.7.0): Used for interpolation and mathematical functions.
- pylinac (3.7.0): Used for additional DICOM image handling.

## Log Files

Log files are stored in the `logs` directory with a date-based naming format.
