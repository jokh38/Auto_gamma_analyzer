"""
This module provides standardized loader functions for various dose file formats.
Each loader is responsible for handling the specifics of its file type and
returning a consistent `StandardDoseData` object.
"""
import numpy as np
import pydicom
from scipy.interpolate import griddata

from .standard_data_model import StandardDoseData
from .utils import logger


def load_dcm(filename: str) -> StandardDoseData:
    """
    Reads a DICOM RT Dose file and converts it into a standard data object.

    This function handles the coordinate system transformation required for DICOM files,
    where the Y-axis is typically inverted. The data grid is flipped, and the
    Y-coordinates are made monotonically increasing to match the standard.

    Args:
        filename (str): The path to the DICOM file.

    Returns:
        StandardDoseData: A standardized data object containing the dose grid,
                          coordinates, and metadata.
    """
    logger.info(f"Loading DICOM file: {filename}")
    try:
        dcm = pydicom.dcmread(filename)

        if dcm.Modality != 'RTDOSE':
            raise ValueError("File is not a DICOM RT Dose file.")

        pixel_data = dcm.pixel_array * dcm.DoseGridScaling
        height, width = pixel_data.shape

        # Extract spacing and origin information
        spacing_x, spacing_y = dcm.PixelSpacing
        pos_x, _, pos_y = dcm.ImagePositionPatient  # DICOM Y is at index 2

        # 1. Create standardized physical coordinates
        # X-coordinates are straightforward (left to right)
        x_coords = (np.arange(width) * spacing_x) + pos_x

        # Y-coordinates need to be transformed to be monotonically increasing (bottom to top)
        # The DICOM standard's ImagePositionPatient refers to the top-left corner.
        # The pixel data is ordered from top to bottom.
        y_coords = (np.arange(height) * spacing_y) + pos_y

        # 2. Flip the data grid to match the ascending Y-coordinates
        # np.flipud flips the array in the up/down direction.
        data_grid = np.flipud(pixel_data)
        # After flipping, the first row of data_grid corresponds to the first y_coord.
        # But the y_coords need to be sorted to be ascending.
        y_coords = np.sort(y_coords)

        # We need to ensure the y-coordinates match the flipped grid.
        # The original pixel_data[0] corresponds to pos_y. After flipud, this data is at the bottom.
        # So the lowest y_coord should be pos_y.
        # Let's re-calculate y_coords to be explicitly ascending from the start.
        # The physical location of rows are: pos_y, pos_y+spacing, ...
        # The data array `pixel_data` has pixel_data[0] corresponding to physical y `pos_y`.
        # We want our `data_grid` to have its y-axis increasing upwards.
        # So, we flip the data grid. The old top row (index 0) becomes the new bottom row.
        # The coordinates must match this new arrangement.
        # The y-coordinate of the first row of the *original* data is `pos_y`.
        # The y-coordinate of the last row of the *original* data is `pos_y + (height-1)*spacing_y`.
        # Our new `y_coords` array should represent the coordinates for the new `data_grid`.
        # The first row of the new `data_grid` (which was the last row of the old one) corresponds to `pos_y + (height-1)*spacing_y`.
        # This means the y_coords should be descending. But the spec says they must be ascending.
        # Let's follow the user's proposal exactly.

        # Reset and follow proposal
        y_start_new = pos_y + (height - 1) * spacing_y
        y_coords_desc = np.arange(y_start_new, pos_y - spacing_y, -spacing_y)
        y_coords = np.flip(y_coords_desc) # now ascending

        # If the length is not correct due to floating point issues, regenerate with linspace
        if len(y_coords) != height:
            y_coords = np.linspace(pos_y, y_start_new, height)


        # 3. Extract metadata
        metadata = {
            "filename": filename,
            "patient_name": str(dcm.get("PatientName", "N/A")),
            "patient_id": dcm.get("PatientID", "N/A"),
            "institution": dcm.get("InstitutionName", "N/A"),
            "file_type": "DICOM",
            "dose_grid_scaling": dcm.DoseGridScaling,
            "image_position_patient": dcm.ImagePositionPatient,
            "pixel_spacing": dcm.PixelSpacing,
        }

        # 4. Create and return the standard data object
        return StandardDoseData(data_grid, x_coords, y_coords, metadata)

    except Exception as e:
        logger.error(f"Failed to load DICOM file {filename}: {e}", exc_info=True)
        raise

# Helper functions for MCC loading, adapted from file_handlers.py
def _detect_mcc_device_type(content: str) -> tuple[int, int]:
    """Detects device and task type from MCC file content."""
    try:
        is_1500 = "SCAN_DEVICE=OCTAVIUS_1500_XDR" in content
        is_merged = "SCAN_OFFAXIS_CROSSPLANE=0.00" in content
        return (2 if is_1500 else 1), (2 if is_merged else 1)
    except Exception as e:
        logger.error(f"Device type detection error: {str(e)}")
        raise

def _extract_mcc_data(lines: list[str], device_type: int) -> np.ndarray:
    """Extracts the dose matrix from the lines of an MCC file."""
    try:
        scan_data_blocks = []
        in_data_block = False
        for line in lines:
            if "BEGIN_DATA" in line:
                in_data_block = True
                current_block = []
                continue
            if "END_DATA" in line:
                in_data_block = False
                if current_block:
                    scan_data_blocks.append(current_block)
                continue
            if in_data_block:
                current_block.append(line)

        n_rows = len(scan_data_blocks)
        if n_rows == 0:
            raise ValueError("No data blocks found in MCC file.")

        # Assuming a square matrix based on number of rows
        matrix = np.full((n_rows, n_rows), -1.0)

        # For 1500 device (staggered), data is arranged in a checkerboard pattern
        if device_type == 2:
            for j, block in enumerate(scan_data_blocks):
                scan_values = [float(line.strip().split()[1]) for line in block if line.strip()]
                # Even rows get data in even columns, odd rows in odd columns
                start_col = j % 2
                for k, value in enumerate(scan_values):
                    col_idx = start_col + 2 * k
                    if col_idx < n_rows:
                        matrix[j, col_idx] = value
        else:  # For 729 device, data is dense in each row
            for j, block in enumerate(scan_data_blocks):
                scan_values = [float(line.strip().split()[1]) for line in block if line.strip()]
                num_values = len(scan_values)
                if num_values > n_rows:
                    logger.warning(f"Row {j} has {num_values} values, but matrix width is {n_rows}. Truncating.")
                    scan_values = scan_values[:n_rows]
                matrix[j, :num_values] = scan_values

        return matrix
    except Exception as e:
        logger.error(f"MCC data extraction error: {str(e)}", exc_info=True)
        raise

def load_mcc(filename: str, target_resolution: float = 2.0) -> StandardDoseData:
    """
    Reads an MCC file, interpolates its sparse data to a dense grid,
    and converts it into a standard data object.

    Args:
        filename (str): The path to the MCC file.
        target_resolution (float): The desired resolution (in mm) for the
                                   interpolated dense grid. Defaults to 2.0.

    Returns:
        StandardDoseData: A standardized data object.
    """
    logger.info(f"Loading MCC file: {filename}")
    try:
        with open(filename, "r") as file:
            content = file.read()
            lines = content.splitlines()

        # 1. Parse file to get raw sparse data
        device_type, task_type = _detect_mcc_device_type(content)
        raw_matrix = _extract_mcc_data(lines, device_type)
        height, width = raw_matrix.shape

        # 2. Define physical coordinates for the raw sparse grid
        if device_type == 2:  # 1500
            origin_x_idx, origin_y_idx, spacing = 26, 26, 5.0
        else:  # 729
            origin_x_idx, origin_y_idx, spacing = 13, 13, 10.0

        # As per proposal, Y-axis increases upwards, so no negation needed
        x_coords_raw = (np.arange(width) - origin_x_idx) * spacing
        y_coords_raw = (np.arange(height) - origin_y_idx) * spacing

        # 3. Convert sparse data to a dense grid via interpolation
        valid_points_indices = np.where(raw_matrix >= 0)
        if valid_points_indices[0].size == 0:
            raise ValueError("MCC file contains no valid data points (>= 0).")

        valid_values = raw_matrix[valid_points_indices]

        # Get the physical coordinates of the valid data points
        phys_points = np.vstack((
            x_coords_raw[valid_points_indices[1]],
            y_coords_raw[valid_points_indices[0]]
        )).T

        # Create a new, regular grid for interpolation
        x_min, x_max = x_coords_raw.min(), x_coords_raw.max()
        y_min, y_max = y_coords_raw.min(), y_coords_raw.max()

        x_coords_new = np.arange(x_min, x_max + target_resolution, target_resolution)
        y_coords_new = np.arange(y_min, y_max + target_resolution, target_resolution)

        grid_x, grid_y = np.meshgrid(x_coords_new, y_coords_new)

        # Perform interpolation
        data_grid = griddata(phys_points, valid_values, (grid_x, grid_y), method='cubic', fill_value=0.0)

        # 4. Extract metadata
        device_name = "OCTAVIUS " + ("1500" if device_type == 2 else "729")
        metadata = {
            "filename": filename,
            "device": device_name,
            "task_type": "merged" if task_type == 2 else "non-merged",
            "file_type": "MCC",
            "original_spacing": spacing,
            "original_points": {'coords': phys_points, 'values': valid_values}
        }

        # 5. Create and return the standard data object
        return StandardDoseData(data_grid, x_coords_new, y_coords_new, metadata)

    except Exception as e:
        logger.error(f"Failed to load MCC file {filename}: {e}", exc_info=True)
        raise
