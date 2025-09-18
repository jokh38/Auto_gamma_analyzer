"""
This module provides classes for handling DICOM and MCC files.
"""
import os
import numpy as np
import yaml
import sys
import csv
import pydicom
from scipy.interpolate import griddata
from utils import logger



class BaseFileHandler:
    """Base class for various file handlers."""
    def __init__(self):
        """Initializes the BaseFileHandler and loads configuration from config.yaml."""
        self.filename = None
        self.pixel_data = None
        self.phys_x_mesh = None
        self.phys_y_mesh = None
        self.physical_extent = None
        self.origin_x = 0
        self.origin_y = 0
        self.pixel_spacing = 1.0
        self.dose_bounds = None
        self.crop_pixel_offset = (0, 0)
            
        # Load configuration
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            self.dta = config.get("dta", 3)
            self.dd = config.get("dd", 3)
            self.suppression_level = config.get("suppression_level", 10)
            self.roi_margin = config.get("roi_margin", 2)
            logger.info(f"Loaded configuration: dta={self.dta}, dd={self.dd}, suppression_level={self.suppression_level}, roi_margin={self.roi_margin}")
        except FileNotFoundError:
            logger.error("Error: config.yaml not found. Please create it.")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error: Could not parse config.yaml. Please check its format: {e}")
            sys.exit(1)
        
    def get_filename(self):
        """Returns the filename."""
        if self.filename:
            return os.path.basename(self.filename)
        return None
    
    def get_physical_extent(self):
        """Returns the physical coordinate range."""
        return self.physical_extent
    
    def get_origin_coords(self):
        """Returns the origin coordinates."""
        return self.origin_x, self.origin_y
        
    def get_spacing(self):
        """Returns the pixel spacing."""
        return self.pixel_spacing, self.pixel_spacing
    
    def get_pixel_data(self):
        """Returns the pixel data."""
        return self.pixel_data
    
    def create_physical_coordinates_dcm(self):
        """Creates the physical coordinate system (abstract method)."""
        raise NotImplementedError("Must be implemented in subclass")
    
    def create_physical_coordinates_mcc(self):
        """Creates the physical coordinate system (abstract method)."""
        raise NotImplementedError("Must be implemented in subclass")
    
    def physical_to_pixel_coord(self, phys_x, phys_y):
        """Converts physical coordinates (mm) to pixel coordinates (abstract method)."""
        raise NotImplementedError("Must be implemented in subclass")
    
    def pixel_to_physical_coord(self, pixel_x, pixel_y):
        """Converts pixel coordinates to physical coordinates (mm) (abstract method)."""
        raise NotImplementedError("Must be implemented in subclass")

    def _calculate_bounds_from_mask(self, mask, margin_mm=0):
        """
        Calculates the bounding box in physical coordinates from a boolean mask.

        Args:
            mask (np.ndarray): A boolean array where True indicates a region of interest.
            margin_mm (int, optional): A margin in millimeters to add to the bounding box. Defaults to 0.

        Returns:
            dict: A dictionary containing the min/max physical coordinates of the bounding box.
        """
        if not np.any(mask):
            return None

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]

        logger.info(f"Dose area pixel bounds: row=({min_row}, {max_row}), col=({min_col}, {max_col})")

        # The y-axis is inverted in physical space relative to pixel space.
        # min_row (top of image) corresponds to max physical y.
        # max_row (bottom of image) corresponds to min physical y.
        min_phys_x, phys_y_for_min_row = self.pixel_to_physical_coord(min_col, min_row)
        max_phys_x, phys_y_for_max_row = self.pixel_to_physical_coord(max_col, max_row)

        # Ensure min_phys_y is always less than max_phys_y for consistency
        min_phys_y = min(phys_y_for_min_row, phys_y_for_max_row)
        max_phys_y = max(phys_y_for_min_row, phys_y_for_max_row)

        logger.info(f"Physical coordinates of pixel bounds (before margin): min=({min_phys_x:.2f}, {min_phys_y:.2f}), max=({max_phys_x:.2f}, {max_phys_y:.2f})")

        if margin_mm > 0:
            min_phys_x -= margin_mm
            max_phys_x += margin_mm
            min_phys_y -= margin_mm
            max_phys_y += margin_mm

        bounds = {
            'min_x': min_phys_x, 'max_x': max_phys_x,
            'min_y': min_phys_y, 'max_y': max_phys_y
        }
        logger.info(f"Calculated dose bounds (physical coords, {margin_mm}mm margin included): {bounds}")
        return bounds

    def calculate_dose_bounds(self, image_data=None, threshold_percent=0, margin_mm=0):
        """Calculates the bounds for a dose area based on a threshold or non-zero dose."""
        if image_data is None:
            image_data = self.get_pixel_data()
        if image_data is None:
            return None

        if threshold_percent > 0:
            # For MCC, -1 values are invalid. This filter is safe for DICOM too.
            valid_data = image_data[image_data >= 0]
            if valid_data.size == 0:
                return None # Cannot determine bounds if no valid data
            
            max_dose = np.max(valid_data)
            if max_dose > 0:
                threshold_val = (threshold_percent / 100.0) * max_dose
                mask = image_data >= threshold_val
            else:
                # If max dose is 0, all valid points are 0.
                # The mask should include these points.
                mask = image_data >= 0
        else:
            # If no threshold, consider all points with dose > 0
            mask = image_data > 0
        
        return self._calculate_bounds_from_mask(mask, margin_mm)
        
    def open_file(self, filename):
        """
        Loads a file (abstract method).

        Args:
            filename (str): The path to the file to load.

        Returns:
            tuple: A tuple containing a boolean success flag and an error message string.
                   (True, None) on success, (False, "error message") on failure.
        """
        try:
            self.filename = filename
            # File loading logic (to be implemented in subclasses)
            return True, None
        except Exception as e:
            error_msg = f"File loading error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


class DicomFileHandler(BaseFileHandler):
    """Class for handling DICOM RT dose files."""
    def __init__(self):
        """Initializes the DicomFileHandler."""
        super().__init__()
        self.dicom_data = None
        self.dicom_origin_x = 0
        self.dicom_origin_y = 0
        self.pixel_spacing = 1.0
        
    def open_file(self, filename):
        """
        Loads and processes a DICOM RT dose file.

        This method reads a DICOM file, extracts dose and patient information,
        and automatically crops the data to the region of interest based on dose levels.

        Args:
            filename (str): The path to the DICOM file.

        Returns:
            tuple: A tuple containing a boolean success flag and an error message string.
                   (True, None) on success, (False, "error message") on failure.
        """
        try:
            self.filename = filename
            self.dicom_data = pydicom.dcmread(filename)
            
            if self.dicom_data.Modality != 'RTDOSE':
                raise ValueError("Selected file is not an RT Dose file")
                
            self.pixel_data = self.dicom_data.pixel_array * self.dicom_data.DoseGridScaling
            
            if hasattr(self.dicom_data, 'PixelSpacing'):
                self.pixel_spacing = float(self.dicom_data.PixelSpacing[0])
            
            height, width = self.pixel_data.shape
            
            if hasattr(self.dicom_data, 'ImagePositionPatient'):
                # ImagePositionPatient: [x, z, y]
                # Calculate pixel origin by dividing physical position by pixel spacing
                self.dicom_origin_x = int(round(self.dicom_data.ImagePositionPatient[0] / self.pixel_spacing)) + 1
                self.dicom_origin_y = int(round(self.dicom_data.ImagePositionPatient[2] / self.pixel_spacing)) - 1 # y-coordinate is at index 2
                
                logger.info(f"Used ImagePositionPatient values: x={self.dicom_data.ImagePositionPatient[0]}, y={self.dicom_data.ImagePositionPatient[2]}")
                logger.info(f"DICOM pixel origin set (from ImagePositionPatient): x={self.dicom_origin_x}, y={self.dicom_origin_y}")
                logger.info(f"Used PixelSpacing value: {self.pixel_spacing}")
            else:
                self.dicom_origin_x = -width // 2 + 1
                self.dicom_origin_y = -height // 2 - 1
                logger.warning(f"DICOM origin info not found. Defaulting to image center: ({self.dicom_origin_x}, {self.dicom_origin_y})")
                
            self.create_physical_coordinates_dcm()

            # Preserve original data and coordinates
            full_pixel_data = self.pixel_data
            full_phys_x_mesh = self.phys_x_mesh
            full_phys_y_mesh = self.phys_y_mesh

            # Auto-crop ROI to area with >1% of max dose + 2cm margin
            self.dose_bounds = self.calculate_dose_bounds(threshold_percent=1, margin_mm=self.roi_margin)

            if self.dose_bounds:
                bounds = self.dose_bounds
                # Convert physical bounds back to pixel coordinates for cropping.
                # Note: At this point, self.physical_to_pixel_coord returns full-grid pixel coordinates
                # because self.crop_pixel_offset is (0, 0).
                min_px, py_for_min_y = self.physical_to_pixel_coord(bounds['min_x'], bounds['min_y'])
                max_px, py_for_max_y = self.physical_to_pixel_coord(bounds['max_x'], bounds['max_y'])

                # Because of y-axis inversion, min physical y corresponds to max pixel y, so we swap.
                min_py = py_for_max_y
                max_py = py_for_min_y
                
                if min_py > max_py: min_py, max_py = max_py, min_py
                if min_px > max_px: min_px, max_px = max_px, min_px

                h, w = full_pixel_data.shape
                min_py, max_py = max(0, min_py), min(h, max_py + 1)
                min_px, max_px = max(0, min_px), min(w, max_px + 1)

                self.pixel_data = full_pixel_data[min_py:max_py, min_px:max_px]
                self.phys_x_mesh = full_phys_x_mesh[min_py:max_py, min_px:max_px]
                self.phys_y_mesh = full_phys_y_mesh[min_py:max_py, min_px:max_px]
                self.crop_pixel_offset = (min_px, min_py)
                self.physical_extent = [self.phys_x_mesh.min(), self.phys_x_mesh.max(), self.phys_y_mesh.min(), self.phys_y_mesh.max()]
                logger.info(f"DICOM data has been cropped to ROI. New shape: {self.pixel_data.shape}")
            
            return True, None
            
        except Exception as e:
            error_msg = f"DICOM file loading error: {e}"
            logger.error(error_msg)
            return False, error_msg


    def physical_to_pixel_coord(self, phys_x, phys_y):
        """Converts physical coordinates (mm) to cropped pixel coordinates."""
        full_grid_px = phys_x / self.pixel_spacing - self.dicom_origin_x
        full_grid_py = -phys_y / self.pixel_spacing - self.dicom_origin_y
        cropped_px = int(round(full_grid_px - self.crop_pixel_offset[0]))
        cropped_py = int(round(full_grid_py - self.crop_pixel_offset[1]))
        
        return cropped_px, cropped_py
    
    def pixel_to_physical_coord(self, pixel_x, pixel_y):
        """Converts cropped pixel coordinates to physical coordinates (mm)."""
        full_grid_px = pixel_x + self.crop_pixel_offset[0]
        full_grid_py = pixel_y + self.crop_pixel_offset[1]
        
        phys_x = (full_grid_px + self.dicom_origin_x) * self.pixel_spacing
        phys_y = (full_grid_py + self.dicom_origin_y) * -self.pixel_spacing
        return phys_x, phys_y

    def get_origin_coords(self):
        """Returns the DICOM origin coordinates in pixels."""
        return self.dicom_origin_x, self.dicom_origin_y

    def get_spacing(self):
        """Returns the DICOM pixel spacing in mm."""
        return self.pixel_spacing, self.pixel_spacing

    def get_patient_info(self):
        """Returns patient information (institution name, ID, name)."""
        if self.dicom_data is None:
            return None, None, None

        institution = self.dicom_data.get("InstitutionName", "N/A")
        patient_id = self.dicom_data.get("PatientID", "N/A")
        patient_name = self.dicom_data.get("PatientName", "N/A")

        return institution, patient_id, patient_name

    def create_physical_coordinates_dcm(self):
        """Creates physical coordinate meshes based on DICOM metadata."""
        if self.pixel_data is None: return
        height, width = self.pixel_data.shape
        phys_x = (np.arange(width) + self.dicom_origin_x) * self.pixel_spacing
        phys_y = (np.arange(height) + self.dicom_origin_y) * -self.pixel_spacing
        self.phys_x_mesh, self.phys_y_mesh = np.meshgrid(phys_x, phys_y)
        # physical_extent의 y축 순서를 min, max로 표준화합니다.
        self.physical_extent = [phys_x.min(), phys_x.max(), phys_y.min(), phys_y.max()]


class MCCFileHandler(BaseFileHandler):
    """Class for handling MCC files."""
    def __init__(self):
        """Initializes the MCCFileHandler."""
        super().__init__()
        self.matrix_data = None
        self.device_type = None
        self.task_type = None
        self.n_rows = None
        self.mcc_origin_x = 0
        self.mcc_origin_y = 0
        self.mcc_spacing_x = 1.0
        self.mcc_spacing_y = 1.0
                
    def get_matrix_data(self):
        """Returns the raw MCC matrix data."""
        return self.matrix_data

    def get_interpolated_matrix_data(self, method='cubic'):
        """
        Returns interpolated MCC matrix data.

        Args:
            method (str, optional): The interpolation method to use ('linear', 'cubic', etc.). Defaults to 'cubic'.

        Returns:
            np.ndarray: The interpolated data, or the original data if interpolation is not possible.
        """
        if self.matrix_data is None: return None
        data = self.matrix_data.copy()
        data[data < 0] = np.nan
        valid_points_indices = np.where(~np.isnan(data))
        if len(valid_points_indices[0]) < 4: return self.matrix_data 
        grid_y, grid_x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        interpolated_data = griddata(
            np.array(list(zip(valid_points_indices[0], valid_points_indices[1]))),
            data[valid_points_indices],
            (grid_y, grid_x),
            method=method
        )
        return interpolated_data        
    
    def open_file(self, filename):
        """
        Loads and processes an MCC file.

        This method reads an MCC file, detects the device type, extracts the dose
        matrix, and sets up the physical coordinate system.

        Args:
            filename (str): The path to the MCC file.

        Returns:
            tuple: A tuple containing a boolean success flag and an error message string.
                   (True, None) on success, (False, "error message") on failure.
        """
        try:
            self.filename = filename
            with open(filename, "r") as file:
                content = file.read()
                lines = content.splitlines()
            
            self.device_type, self.task_type = self.detect_device_type(content)
            
            N_begin = content.count("BEGIN_DATA")
            self.n_rows = N_begin
            self.matrix_data = self.extract_data(lines, N_begin, self.device_type, self.task_type)
            self.pixel_data = self.matrix_data
            
            self._set_device_parameters()
            self.create_physical_coordinates_mcc()
            
            logger.info(f"MCC file loaded successfully: {self.get_device_name()}")
            return True, None
                        
        except Exception as e:
            error_msg = f"File open error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def crop_to_bounds(self, bounds):
        """
        Crops the MCC data based on physical coordinate bounds from DICOM.
        """
        if self.matrix_data is None or bounds is None:
            return

        full_matrix_data = self.matrix_data
        full_phys_x_mesh = self.phys_x_mesh
        full_phys_y_mesh = self.phys_y_mesh

        min_px, max_py_from_min_y = self.physical_to_pixel_coord(bounds['min_x'], bounds['min_y'])
        max_px, min_py_from_max_y = self.physical_to_pixel_coord(bounds['max_x'], bounds['max_y'])

        min_py, max_py = min_py_from_max_y, max_py_from_min_y

        if min_py > max_py: min_py, max_py = max_py, min_py
        if min_px > max_px: min_px, max_px = max_px, min_px

        h, w = full_matrix_data.shape
        min_py, max_py = max(0, min_py), min(h, max_py + 1)
        min_px, max_px = max(0, min_px), min(w, max_px + 1)

        self.matrix_data = full_matrix_data[min_py:max_py, min_px:max_px]
        self.pixel_data = self.matrix_data
        self.phys_x_mesh = full_phys_x_mesh[min_py:max_py, min_px:max_px]
        self.phys_y_mesh = full_phys_y_mesh[min_py:max_py, min_px:max_px]
        self.crop_pixel_offset = (min_px, min_py)
        self.physical_extent = [self.phys_x_mesh.min(), self.phys_x_mesh.max(), self.phys_y_mesh.min(), self.phys_y_mesh.max()]
        
        logger.info(f"MCC data has been cropped to DICOM ROI. New shape: {self.matrix_data.shape}")

    def _set_device_parameters(self):
        """Sets the origin and spacing parameters based on the detected device type."""
        if self.device_type == 2:  # 1500
            self.mcc_origin_x = 26  # 0-based index
            self.mcc_origin_y = 26  # 0-based index
            self.mcc_spacing_x = 5.0
            self.mcc_spacing_y = 5.0
        else:  # 725
            self.mcc_origin_x = 13  # 0-based index
            self.mcc_origin_y = 13  # 0-based index
            self.mcc_spacing_x = 10.0
            self.mcc_spacing_y = 10.0
    
    def extract_data(self, lines, N_begin, device_type, task_type):
        """
        Extracts the dose matrix from the lines of an MCC file.

        Args:
            lines (list): The lines of the MCC file.
            N_begin (int): The number of data blocks (rows).
            device_type (int): The type of the device (1 for 725, 2 for 1500).
            task_type (int): The type of task (1 for non-merged, 2 for merged).

        Returns:
            np.ndarray: The extracted dose matrix.
        """
        try:
            scan_data_blocks = []
            in_data_block = False
            current_block = []
            for line in lines:
                if "BEGIN_DATA" in line:
                    in_data_block = True
                    current_block = []
                    continue
                if "END_DATA" in line:
                    in_data_block = False
                    if current_block:
                        scan_data_blocks.append(current_block)
                    current_block = []
                    continue
                if in_data_block:
                    current_block.append(line)

            if len(scan_data_blocks) != N_begin:
                logger.warning(f"Mismatched scan blocks. Expected {N_begin}, found {len(scan_data_blocks)}.")
                N_begin = len(scan_data_blocks)

            matrix_octavius_mat_tmp = np.full((N_begin, N_begin), -1.0)

            if device_type == 2 and task_type == 1:  # OCTAVIUS 1500 XDR, non-merged (staggered)
                for j, block in enumerate(scan_data_blocks):
                    scan_values = [float(line.strip().split()[1]) for line in block if line.strip()]
                    if j % 2 == 0:  # Even rows (0, 2, ...) get data in even columns
                        for k, value in enumerate(scan_values):
                            col_idx = 2 * k
                            if col_idx < N_begin:
                                matrix_octavius_mat_tmp[j, col_idx] = value
                    else:  # Odd rows (1, 3, ...) get data in odd columns
                        for k, value in enumerate(scan_values):
                            col_idx = 2 * k + 1
                            if col_idx < N_begin:
                                matrix_octavius_mat_tmp[j, col_idx] = value
            else:  # Fallback for other devices or merged data (assumed non-staggered)
                for j, block in enumerate(scan_data_blocks):
                    scan_values = [float(line.strip().split()[1]) for line in block if line.strip()]
                    for k, value in enumerate(scan_values):
                        if k < N_begin:
                            matrix_octavius_mat_tmp[j, k] = value
            return matrix_octavius_mat_tmp
        except Exception as e:
            logger.error(f"Data extraction error: {str(e)}", exc_info=True)
            raise
        
    def detect_device_type(self, content):
        """
        Detects the device type and task type from the MCC file content.

        Args:
            content (str): The content of the MCC file.

        Returns:
            tuple: A tuple containing the device type (int) and task type (int).
        """
        try:
            is_1500 = "SCAN_DEVICE=OCTAVIUS_1500_XDR" in content
            is_merged = "SCAN_OFFAXIS_CROSSPLANE=0.00" in content
            return (2 if is_1500 else 1), (2 if is_merged else 1)
        except Exception as e:
            logger.error(f"Device type detection error: {str(e)}")
            raise

    def get_device_name(self):
        """Returns the name of the detected device."""
        if self.device_type == 2: return "OCTAVIUS 1500" + (" with merge" if self.task_type == 2 else "")
        else: return "OCTAVIUS 725" + (" with merge" if self.task_type == 2 else "")
                
    def get_origin_coords(self):
        """Returns the MCC origin coordinates in pixels."""
        return self.mcc_origin_x, self.mcc_origin_y
            
    def get_spacing(self):
        """Returns the MCC pixel spacing in mm."""
        return self.mcc_spacing_x, self.mcc_spacing_y

    def create_physical_coordinates_mcc(self):
        """Creates physical coordinate meshes for the MCC data."""
        if self.matrix_data is None: return
        height, width = self.matrix_data.shape
        phys_x = (np.arange(width) - self.mcc_origin_x) * self.mcc_spacing_x
        phys_y = -(np.arange(height) - self.mcc_origin_y) * self.mcc_spacing_y
        self.phys_x_mesh, self.phys_y_mesh = np.meshgrid(phys_x, phys_y)
        self.physical_extent = [phys_x.min(), phys_x.max(), phys_y.min(), phys_y.max()]
            
    def physical_to_pixel_coord(self, phys_x, phys_y):
        """Converts physical coordinates (mm) to cropped pixel coordinates."""
        full_grid_px = phys_x / self.mcc_spacing_x + self.mcc_origin_x
        full_grid_py = -phys_y / self.mcc_spacing_y + self.mcc_origin_y

        cropped_px = int(round(full_grid_px - self.crop_pixel_offset[0]))
        cropped_py = int(round(full_grid_py - self.crop_pixel_offset[1]))

        return cropped_px, cropped_py
    
    def pixel_to_physical_coord(self, pixel_x, pixel_y):
        """Converts cropped pixel coordinates to physical coordinates (mm)."""
        full_grid_px = pixel_x + self.crop_pixel_offset[0]
        full_grid_py = pixel_y + self.crop_pixel_offset[1]

        phys_x = (full_grid_px - self.mcc_origin_x) * self.mcc_spacing_x
        phys_y = -(full_grid_py - self.mcc_origin_y) * self.mcc_spacing_y
        return phys_x, phys_y
