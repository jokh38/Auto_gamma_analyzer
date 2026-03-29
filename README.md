# Auto Gamma Analysis Tool

Desktop application for 2D gamma analysis between DICOM RT Dose files and MCC measurement files.

## Features

- Load DICOM RT Dose and MCC files
- Support OCTAVIUS 725 and OCTAVIUS 1500 MCC formats
- Interactive vertical and horizontal profile views
- Adjustable gamma criteria (`DTA`, `DD`, `Global`/`Local`)
- PDF/JPEG report generation
- Batch `Auto analysis` by beam number
- `Clear data` to unload File A and File B without closing the app

## Beam Matching In Auto Analysis

`Auto analysis` scans a selected directory and groups files by beam number using the filename pattern `(\d+)G`.

Examples:

- `1G180.dcm` -> beam `1`
- `2G220_2cm.mcc` -> beam `2`
- `3G280_2cm.dcm` -> beam `3`

Rules:

- Only `.dcm` files with DICOM header `Modality == RTDOSE` are used
- DICOM RT Plan files are skipped
- Each beam must have one RT Dose `.dcm` and one `.mcc`
- Reports are saved in the same directory selected for auto analysis

## Requirements

- Python 3.12 recommended
- PyQt5
- numpy
- matplotlib
- pydicom
- scipy
- PyYAML

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run From Source

Start the GUI:

```bash
python run.py
```

On first file load, the dialog starts at `C:\` on Windows. After a directory is used once, the app remembers it with `QSettings`.

## Build With PyInstaller

This repository already includes the PyInstaller spec file:

- [AutoGammaAnalyzer.spec](/mnt/c/MOQUI_SMC/Auto_gamma_analyzer/AutoGammaAnalyzer.spec)

The existing build environment used in this project is:

- `.build-env`

Build command:

```bash
'.build-env/Scripts/python.exe' -m PyInstaller --clean --noconfirm AutoGammaAnalyzer.spec
```

The built executable is generated at:

- [dist/AutoGammaAnalyzer.exe](/mnt/c/MOQUI_SMC/Auto_gamma_analyzer/dist/AutoGammaAnalyzer.exe)

## Main Files

- [run.py](/mnt/c/MOQUI_SMC/Auto_gamma_analyzer/run.py): application entry point
- [src/main_app.py](/mnt/c/MOQUI_SMC/Auto_gamma_analyzer/src/main_app.py): main window and UI wiring
- [src/app_controller.py](/mnt/c/MOQUI_SMC/Auto_gamma_analyzer/src/app_controller.py): application logic
- [src/file_handlers.py](/mnt/c/MOQUI_SMC/Auto_gamma_analyzer/src/file_handlers.py): DICOM and MCC handlers
- [src/analysis.py](/mnt/c/MOQUI_SMC/Auto_gamma_analyzer/src/analysis.py): profile extraction and gamma analysis
- [src/reporting.py](/mnt/c/MOQUI_SMC/Auto_gamma_analyzer/src/reporting.py): report generation
- [src/ui_components.py](/mnt/c/MOQUI_SMC/Auto_gamma_analyzer/src/ui_components.py): plotting and UI components

## Notes

- Default gamma criteria are loaded from `config.yaml`
- Log files are written to the user data directory
- Auto analysis disables the main UI while batch processing is running and shows a small progress window
