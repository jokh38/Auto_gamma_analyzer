# Auto Gamma Analysis Tool

2D Gamma Analysis Tool for comparing DICOM RT dose files with MCC measurement data.

## Features

- DICOM RT dose file loader
- MCC file loader (supports both OCTAVIUS 725 and OCTAVIUS 1500 formats)
- Interactive dose profiles (vertical and horizontal)
- 2D Gamma analysis with adjustable parameters
- PDF report generation

## Prerequisites

- Python 3.6+
- PyQt5
- numpy
- matplotlib
- pydicom
- pymedphys
- pylinac

## File Structure

- `main_app.py`: Main application class and program entry point
- `file_handlers.py`: Classes for handling DICOM and MCC files
- `ui_components.py`: UI component classes and drawing functions
- `analysis.py`: Profile extraction and gamma analysis functions
- `utils.py`: Utility functions and logging setup

## Usage

1. Run the application:
```
python main_app.py
```

2. Load a DICOM RT dose file and an MCC measurement file.
3. Adjust the origin if necessary.
4. Click on the DICOM image to generate a profile.
5. Set gamma analysis parameters and run the analysis.
6. Generate a PDF report of the results.

## Library Reference

- numpy (1.21.0): Used for array operations and calculations
- matplotlib (3.5.1): Used for visualization and plotting
- PyQt5 (5.15.6): Used for the graphical user interface
- pydicom (2.3.0): Used for loading and handling DICOM files
- pymedphys (0.36.0): Used for gamma analysis calculations
- pylinac (3.7.0): Used for additional DICOM image handling

## Log Files

Log files are stored in the `logs` directory with a date-based naming format.

## Notes

- MCC files from both OCTAVIUS 725 and OCTAVIUS 1500 devices are supported
- Default gamma criteria are 3mm/3%
- Both local and global gamma analysis methods are supported
