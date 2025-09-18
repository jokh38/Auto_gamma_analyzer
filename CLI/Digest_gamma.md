This document provides an overview of the project based on the docstrings found in the Python source code.

---

## 1. `src/analysis.py`

This module provides functions for analyzing and comparing DICOM and MCC data, including profile extraction and gamma analysis.

### Functions

-   **`extract_profile_data(direction, fixed_position, dicom_handler, mcc_handler=None)`**
    
    Extracts profile data from DICOM and MCC datasets.
    
-   **`perform_gamma_analysis(reference_handler, evaluation_handler, dose_percent_threshold, distance_mm_threshold, global_normalisation=True, threshold=10, max_gamma=None, save_csv=False, csv_dir=None)`**
    
    Performs gamma index analysis directly between sparse reference points (MCC) and a dense evaluation grid (DICOM) without interpolation.
    

---

## 2. `src/file_handlers.py`

This module provides classes for handling DICOM and MCC files.

### Classes

#### `BaseFileHandler`

Base class for various file handlers.

-   **`__init__(self)`**
    
    Initializes the BaseFileHandler and loads configuration from config.yaml.
    
-   **`get_filename(self)`**
    
    Returns the filename.
    
-   **`get_physical_extent(self)`**
    
    Returns the physical coordinate range.
    
-   **`get_origin_coords(self)`**
    
    Returns the origin coordinates.
    
-   **`get_spacing(self)`**
    
    Returns the pixel spacing.
    
-   **`get_pixel_data(self)`**
    
    Returns the pixel data.
    
-   **`create_physical_coordinates_dcm(self)`**
    
    Creates the physical coordinate system (abstract method).
    
-   **`create_physical_coordinates_mcc(self)`**
    
    Creates the physical coordinate system (abstract method).
    
-   **`physical_to_pixel_coord(self, phys_x, phys_y)`**
    
    Converts physical coordinates (mm) to pixel coordinates (abstract method).
    
-   **`pixel_to_physical_coord(self, pixel_x, pixel_y)`**
    
    Converts pixel coordinates to physical coordinates (mm) (abstract method).
    
-   **`_calculate_bounds_from_mask(self, mask, margin_mm=0)`**
    
    Calculates the bounding box in physical coordinates from a boolean mask.
    
-   **`calculate_dose_bounds(self, image_data=None, threshold_percent=0, margin_mm=0)`**
    
    Calculates the bounds for a dose area based on a threshold or non-zero dose.
    
-   **`open_file(self, filename)`**
    
    Loads a file (abstract method).
    

#### `DicomFileHandler(BaseFileHandler)`

Class for handling DICOM RT dose files.

-   **`__init__(self)`**
    
    Initializes the DicomFileHandler.
    
-   **`open_file(self, filename)`**
    
    Loads and processes a DICOM RT dose file.
    
-   **`physical_to_pixel_coord(self, phys_x, phys_y)`**
    
    Converts physical coordinates (mm) to cropped pixel coordinates.
    
-   **`pixel_to_physical_coord(self, pixel_x, pixel_y)`**
    
    Converts cropped pixel coordinates to physical coordinates (mm).
    
-   **`get_origin_coords(self)`**
    
    Returns the DICOM origin coordinates in pixels.
    
-   **`get_spacing(self)`**
    
    Returns the DICOM pixel spacing in mm.
    
-   **`get_patient_info(self)`**
    
    Returns patient information (institution name, ID, name).
    
-   **`create_physical_coordinates_dcm(self)`**
    
    Creates physical coordinate meshes based on DICOM metadata.
    

#### `MCCFileHandler(BaseFileHandler)`

Class for handling MCC files.

-   **`__init__(self)`**
    
    Initializes the MCCFileHandler.
    
-   **`get_matrix_data(self)`**
    
    Returns the raw MCC matrix data.
    
-   **`get_interpolated_matrix_data(self, method='cubic')`**
    
    Returns interpolated MCC matrix data.
    
-   **`open_file(self, filename)`**
    
    Loads and processes an MCC file.
    
-   **`crop_to_bounds(self, bounds)`**
    
    Crops the MCC data based on physical coordinate bounds from DICOM.
    
-   **`_set_device_parameters(self)`**
    
    Sets the origin and spacing parameters based on the detected device type.
    
-   **`extract_data(self, lines, N_begin, device_type, task_type)`**
    
    Extracts the dose matrix from the lines of an MCC file.
    
-   **`detect_device_type(self, content)`**
    
    Detects the device type and task type from the MCC file content.
    
-   **`get_device_name(self)`**
    
    Returns the name of the detected device.
    
-   **`get_origin_coords(self)`**
    
    Returns the MCC origin coordinates in pixels.
    
-   **`get_spacing(self)`**
    
    Returns the MCC pixel spacing in mm.
    
-   **`create_physical_coordinates_mcc(self)`**
    
    Creates physical coordinate meshes for the MCC data.
    
-   **`physical_to_pixel_coord(self, phys_x, phys_y)`**
    
    Converts physical coordinates (mm) to cropped pixel coordinates.
    
-   **`pixel_to_physical_coord(self, pixel_x, pixel_y)`**
    
    Converts cropped pixel coordinates to physical coordinates (mm).
    

---

## 3. `src/main.py`

This module contains the main function for running the gamma analysis from the command line.

### Functions

-   **`main()`**
    
    Main function to run the gamma analysis from the command line.
    

---

## 4. `src/reporting.py`

This module provides functions for generating reports of the gamma analysis results.

### Functions

-   **`generate_report(output_path, dicom_handler, mcc_handler, gamma_map, gamma_stats, dta, dd, suppression_level, ver_profile_data, hor_profile_data, mcc_interp_data=None, dd_map=None, dta_map=None, dd_stats=None, dta_stats=None, additional_profiles=None)`**
    
    Generates a comprehensive PDF report of the gamma analysis results.
    

---

## 5. `src/utils.py`

This module provides utility functions for the application, including logging and array manipulation.

### Functions

-   **`setup_logger(name)`**
    
    Sets up a logger for the application.
    
-   **`find_nearest_index(array, value)`**
    
    Returns the index of the element in the array that is closest to the given value.
    
-   **`save_map_to_csv(data_map, phys_x_mesh, phys_y_mesh, output_filename)`**
    
    Saves a 2D data map with physical coordinates to a CSV file.
    

---

## 6. `tests/test_analysis.py`

This module contains unit tests for the analysis functions.

### Classes

#### `TestGammaAnalysis(unittest.TestCase)`

Test suite for the gamma analysis functions.

-   **`setUp(self)`**
    
    Set up the test case by loading the data files.
    
-   **`test_perform_gamma_analysis(self)`**
    
    Test the perform\_gamma\_analysis function with example data.