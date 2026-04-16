# Auto Gamma Analyzer

Desktop application for 2D gamma analysis between DICOM RT Dose files and MCC measurement files.

## Features

- Load DICOM RT Dose (File A) and MCC measurement (File B) files
- Supports **OCTAVIUS 725** and **OCTAVIUS 1500 XDR** detector formats
- Auto-detects device type from MCC file content
- Interactive 2D dose display with clickable profile extraction (horizontal / vertical)
- **Data rotation** — rotate File A or File B 90° CW/CCW to align orientations
- Per-file normalization factor control (File A / File B)
- Adjustable DICOM origin offset (Delta X / Delta Y in mm)
- Gamma analysis with configurable criteria:
  - DTA (mm), DD (%), Global or Local normalization
  - Dose suppression threshold (%)
- Interpolation methods: `linear`, `cubic`, `nearest` (configurable in `config.yaml`)
- ROI auto-crop on DICOM load based on dose threshold
- PDF or JPEG report generation with professional single-page layout
- **Batch Auto Analysis** — scan a directory, match files by beam number, generate reports automatically
- `Clear` button to unload files without closing the app

## Beam Matching in Auto Analysis

`Auto analysis` scans a selected directory and groups DICOM and MCC files by beam number using the filename pattern `(\d+)G`.

Examples:

| Filename | Beam |
|---|---|
| `1G180.dcm` | `1` |
| `2G220_2cm.mcc` | `2` |
| `3G280_2cm.dcm` | `3` |

Rules:

- Only `.dcm` files with DICOM `Modality == RTDOSE` are matched (RT Plan files are skipped)
- Each beam must have exactly one RT Dose `.dcm` and one `.mcc`
- Reports are saved in the same directory selected for auto analysis
- The main UI is disabled during batch processing; a progress window shows status

## Configuration

Default gamma criteria and interpolation settings are loaded from `config.yaml`:

```yaml
dta: 2                          # mm
dd: 2                           # %
suppression_level: 5            # % of max dose
roi_margin: 5                   # px margin around ROI
interpolation_method: "cubic"   # linear | cubic | nearest
smoothing_factor: 0             # Gaussian smoothing (0 = off)
fill_value_type: "zero"         # zero | nan
mcc_interpolation_method: "cubic"
roi_threshold_percent: 1        # DICOM ROI auto-crop threshold
```

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

```bash
python run.py
```

File dialog directories are remembered between sessions using `QSettings`.

## Build With PyInstaller

A `.spec` file is included in the repository. The build environment used in this project is `.build-env`.

```bash
'.build-env/Scripts/python.exe' -m PyInstaller --clean --noconfirm AutoGammaAnalyzer.spec
```

Output: `dist/AutoGammaAnalyzer.exe`

## Source Layout

| File | Purpose |
|---|---|
| `run.py` | Application entry point |
| `src/main_app.py` | Main window and UI wiring |
| `src/app_controller.py` | Application logic and event handling |
| `src/data_manager.py` | Shared application state |
| `src/file_handlers.py` | DICOM and MCC file loading |
| `src/analysis.py` | Profile extraction and gamma analysis |
| `src/reporting.py` | PDF/JPEG report generation |
| `src/ui_components.py` | Plotting and UI components |
| `src/plot_styles.py` | Matplotlib style helpers |
| `src/styles.py` | Qt stylesheet |
| `config.yaml` | Default gamma and interpolation settings |
