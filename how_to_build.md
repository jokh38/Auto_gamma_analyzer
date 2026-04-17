# Auto Gamma Analyzer - Build Guide

## Quick Start

```powershell
# 1. Create clean build environment
python -m venv build_venv
build_venv\Scripts\activate
pip install numpy scipy matplotlib PyQt5 pydicom PyYAML pyinstaller

# 2. Build
build_venv\Scripts\pyinstaller.exe --clean --noconfirm AutoGammaAnalyzer.spec

# 3. Verify
Get-Item dist\AutoGammaAnalyzer.exe | Select-Object Name, @{N='Size (MB)';E={[math]::Round($_.Length/1MB, 1)}}
```

Output: `dist\AutoGammaAnalyzer.exe` (~90 MB)

---

## 1. Why a Clean Virtualenv Is Mandatory

The global Python environment has torch (~2 GB), cupy, transformers, scikit-learn, opencv, numba, pandas, sympy, sphinx, IPython installed.

**PyInstaller's `excludes` list blocks Python-level imports but does NOT prevent binary DLL scanning.** PyInstaller scans all `.pyd`/`.dll` files in `site-packages/` and pulls in transitive binary dependencies from torch, cupy, etc., adding hundreds of MB.

| Build Environment | EXE Size | Notes |
|---|---|---|
| Global (dirty, with torch/cupy/etc.) | **1,078 MB** | Binary DLLs leak in despite excludes |
| Clean venv (only 7 packages) | **~91 MB** | 92% reduction, no code changes |

**Always build in a clean venv.** This is the single highest-impact action.

---

## 2. Project Dependencies

### Required (actually used in code)

| Package | Usage | Modules |
|---------|-------|---------|
| numpy | Array operations, calculations | All src/*.py |
| scipy | `interpolate.griddata`, `interpolate.interpn`, `ndimage.gaussian_filter` | analysis.py, load_mcc.py, ui_components.py, file_handlers.py |
| matplotlib | Plotting, colormap, PDF report generation | reporting.py, ui_components.py, plot_styles.py |
| PyQt5 | GUI framework | main_app.py, ui_components.py, app_controller.py |
| pydicom | DICOM RT dose file loading | load_dcm.py, app_controller.py, file_handlers.py |
| PyYAML | config.yaml parsing | utils.py, analysis.py, file_handlers.py, load_mcc.py |
| PyInstaller | Build tool (not a runtime dep) | — |

### Version tested

| Package | Version |
|---------|---------|
| Python | 3.12.4 |
| PyInstaller | 6.19.0 |
| Platform | Windows 11 (10.0.26200) |

### Not required (not imported in src/)

pandas, torch, torchvision, torchaudio, transformers, scikit-learn, scikit-image, opencv, numba, cupy, sympy, IPython, jupyter, sphinx, pytest

---

## 3. Spec File Explained

The `AutoGammaAnalyzer.spec` file contains all build optimizations:

### 3.1 pathex (critical for import discovery)

```python
pathex=['src'],
```

PyInstaller cannot statically analyze imports inside bundled data files. Adding `src/` to `pathex` allows it to trace the full import chain from `run.py` → `main_app.py` → all src modules.

Without this, matplotlib, scipy, yaml, and other imports inside `src/*.py` will not be discovered and the exe will crash at runtime with `ModuleNotFoundError`.

### 3.2 datas

```python
datas=[('config.yaml', '.')],
```

Only `config.yaml` needs to be bundled. The `src/` directory is NOT in `datas` because `pathex` handles import discovery — PyInstaller compiles `src/*.py` as Python modules automatically. (Putting `src/` in `datas` would bundle them as raw data files, which is what caused the original missing-module errors.)

### 3.3 hiddenimports

Required because some imports cannot be discovered statically:

```python
hiddenimports=[
    'PyQt5.sip',                              # PyQt5 C binding
    'pydicom.encoders.gdcm',                   # Optional DICOM codec — WARNING if not found (safe to ignore)
    'pydicom.encoders.pylibjpeg',              # Optional DICOM codec — WARNING if not found (safe to ignore)
    'scipy.special._ufuncs_cxx',               # scipy compiled extension
    'scipy.linalg.cython_blas',                # BLAS interface
    'scipy.linalg.cython_lapack',              # LAPACK interface
    'scipy.integrate',                         # scipy internal dependency
    'scipy.interpolate',                       # griddata, interpn
    'matplotlib',                              # plotting
    'matplotlib.pyplot',                       # pyplot interface
    'matplotlib.backends.backend_pdf',         # PDF report generation
    'matplotlib.colors',                       # colormap normalization
    'matplotlib.patches',                      # report legend patches
    'yaml',                                    # config.yaml parsing
],
```

### 3.4 hooksconfig (matplotlib backend pinning)

```python
hooksconfig={
    'matplotlib': {
        'backends': ['Qt5Agg'],
    },
},
```

Only include the Qt5Agg backend — excludes TkAgg, GTK, WX, MacOSX backends. Saves ~10-30 MB.

### 3.5 excludes

```python
excludes=[
    'torch', 'torchvision', 'torchaudio', 'transformers',
    'tensorflow', 'keras', 'sklearn', 'scikit-learn',
    'cv2', 'opencv-python', 'numba', 'llvmlite',
    'IPython', 'jedi', 'parso', 'sphinx', 'docutils',
    'babel', 'nltk', 'notebook', 'nbformat',
    'pytest', 'py', 'zmq', 'tornado', 'jinja2',
    'cloudpickle', 'sympy', 'nacl',
    'win32com', 'pythoncom', 'pywintypes',
    'soundfile', 'av', 'sqlite3',
    'pandas', 'openpyxl', 'lxml', 'pyreadstat', 'tables', 'pyarrow',
    'PySide2', 'PySide6', 'PyQt6',
],
```

**Do NOT exclude stdlib modules or scipy submodules.** See Section 5 (Pitfalls) for details.

### 3.6 Qt plugin trimming

After the Analysis block, unused Qt plugins are filtered out:

```python
qt_junk = [
    'qtwebengine', 'qtbluetooth', 'qtconnectivity',
    'qtpositioning', 'qtsensors', 'qtserialport', 'qtspeech',
    'qt3d', 'qtgamepad', 'qtremoteobjects', 'qtscript',
    'qtscxml', 'qtvirtualkeyboard', 'qtwebchannel', 'qtwebview',
    'qtxmlpatterns', 'qtactiveqt', 'qthelp', 'qtdesigner',
    'Qt5Bluetooth', 'Qt5Positioning', 'Qt5Sensors',
    'Qt5Multimedia', 'Qt5Xml', 'Qt5Network',
]
a.binaries = [x for x in a.binaries if not any(junk in x[0] for junk in qt_junk)]
```

Also filters imageformats (keep only PNG, JPEG, GIF, ICO, SVG) and platform plugins (keep only qwindows).

---

## 4. Size Breakdown

### Where the ~90 MB goes

| Component | Size | Notes |
|-----------|------|-------|
| PyQt5 (Qt5 DLLs + plugins) | ~25-35 MB | After plugin trimming |
| scipy (compiled Fortran/C libs) | ~25-35 MB | Full scipy (internal deps required) |
| numpy + OpenBLAS | ~10-15 MB | Already native C |
| matplotlib (Qt5Agg backend) | ~10-15 MB | After backend pinning |
| pydicom | ~3-5 MB | Pure Python |
| Python runtime + stdlib | ~5-8 MB | python312.dll + stdlib |
| **Your code (src/*.py)** | **<1 MB** | Negligible |

### What was removed vs the 1,078 MB build

| Source of bloat | Size removed | How |
|---|---|---|
| torch/cupy/transformers DLLs | ~500-600 MB | Clean venv |
| Unused Qt plugins (WebEngine etc.) | ~50-100 MB | Binary filtering |
| Unused matplotlib backends | ~10-20 MB | hooksconfig |
| Other unused packages (sklearn, cv2, etc.) | ~200-300 MB | Clean venv |

---

## 5. Pitfalls & Lessons Learned

### 5.1 Do NOT exclude stdlib modules

```
pydoc   → required by scipy._lib._docscrape
unittest → required by pyparsing.testing
doctest  → may be needed by various libraries
```

These stdlib modules are <1 MB total. Excluding them saves nothing but causes `ModuleNotFoundError` at runtime when scipy/pyparsing tries to import them.

### 5.2 Do NOT exclude scipy submodules

scipy has deep internal cross-dependencies:

```
scipy.interpolate.griddata
  → scipy.spatial (Delaunay triangulation)
    → scipy.sparse (internal dependency)
  → scipy.special (ufuncs)
  → scipy.linalg (BLAS/LAPACK)
```

Excluding `scipy.spatial`, `scipy.sparse`, etc. saves ~5 MB but causes `ModuleNotFoundError` when `griddata` is called. The clean venv already removed the real bloat — scipy submodule excludes are chasing pennies.

### 5.3 src/ must be in pathex, NOT in datas

| Approach | Result |
|---|---|
| `datas=[('src', 'src')]` | src files bundled as raw data → PyInstaller can't trace imports → `ModuleNotFoundError` for matplotlib, scipy, yaml |
| `pathex=['src']` | PyInstaller traces imports correctly → all deps discovered automatically |
| Both together | Works but redundant; `datas` adds unnecessary raw file copies |

### 5.4 The exe must not be running during rebuild

If `dist\AutoGammaAnalyzer.exe` is running, PyInstaller will fail with:

```
PermissionError: [WinError 5] Access is denied: 'dist\AutoGammaAnalyzer.exe'
```

Close the app before rebuilding, or kill it: `Stop-Process -Name AutoGammaAnalyzer -Force`

### 5.5 Safe-to-ignore build warnings

The following warnings appear during analysis but are harmless — the missing packages are optional codecs not needed for core DICOM functionality:

```
ERROR: Hidden import 'pydicom.encoders.gdcm' not found
ERROR: Hidden import 'pydicom.encoders.pylibjpeg' not found
WARNING: Hidden import "scipy.special._cdflib" not found!
```

These entries are kept in `hiddenimports` so the build succeeds automatically if the packages are later installed.

---

## 6. config.yaml

The app reads `config.yaml` at runtime. The spec file bundles it:

```python
datas=[('config.yaml', '.')],
```

Current config:

```yaml
dta: 2
dd: 2
suppression_level: 5
roi_margin: 5
save_csv: false
csv_export_path: "csv_exports"
interpolation_method: "cubic"
smoothing_factor: 0
fill_value_type: "zero"
mcc_interpolation_method: "cubic"
mcc_fill_value_type: "zero"
roi_threshold_percent: 1
```

---

## 7. Alternative: Nuitka (Optional)

Nuitka compiles Python → C → native binary. Benefits over PyInstaller:
- **2-4x faster runtime** performance
- **Harder to reverse-engineer** (compiled C vs PyInstaller's .pyc)
- **Fewer antivirus false positives**

But for this stack the exe size is similar (~80-100 MB) because the size comes from pre-compiled C extension DLLs, not Python bytecode.

```bash
pip install nuitka zstandard ordered-set

python -m nuitka ^
    --mode=onefile ^
    --enable-plugin=pyqt5 ^
    --include-qt-plugins=sensible ^
    --python-flag=no_docstrings ^
    --noinclude-pytest-mode=nofollow ^
    --noinclude-setuptools-mode=nofollow ^
    --noinclude-IPython-mode=nofollow ^
    --noinclude-custom-mode=QtWebEngine:error ^
    --deployment ^
    --lto=yes ^
    --windows-console-mode=disable ^
    --output-dir=dist ^
    --assume-yes-for-downloads ^
    run.py
```

**Caveat**: Nuitka has [known issues with PyQt5](https://nuitka.net/pages/pyqt5.html) (callbacks, threading). For best compatibility, consider migrating to **PySide6**. Build time: 10-30 minutes (vs PyInstaller's ~90 seconds).

---

## 8. Build Environment Checklist

Before building, verify:

- [ ] Python 3.12+ installed
- [ ] Clean virtualenv activated (only 7 required packages + pyinstaller)
- [ ] `numpy.show_config()` shows OpenBLAS (not MKL)
- [ ] `config.yaml` exists in project root
- [ ] `src/` directory exists with all modules
- [ ] No torch/cupy/transformers in `pip list`
- [ ] Previous exe is not running
- [ ] Build command: `build_venv\Scripts\pyinstaller.exe --clean --noconfirm AutoGammaAnalyzer.spec`
- [ ] Expected output: `dist\AutoGammaAnalyzer.exe` (~91 MB)
