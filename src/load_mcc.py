"""
This module provides the standardized loader function for MCC dose file formats.
It handles the specifics of parsing MCC files and returns a consistent
`StandardDoseData` object.
"""
import numpy as np
from scipy.interpolate import griddata

from .standard_data_model import StandardDoseData
from .utils import logger

# Helper functions for MCC loading
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
                continue
            if in_data_block:
                # Ensure line is not empty and can be split
                if line.strip():
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

        # 2. Define physical coordinates for the raw sparse grid (Y-axis increasing upwards)
        if device_type == 2:  # 1500
            origin_x_idx, origin_y_idx, spacing = 26, 26, 5.0
        else:  # 729
            origin_x_idx, origin_y_idx, spacing = 13, 13, 10.0

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
