"""
This module provides the standardized loader function for DICOM RT Dose files.
"""
import numpy as np
import pydicom

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

        # DICOM PixelSpacing is stored as [row_spacing, column_spacing].
        # The in-plane x axis uses the column spacing, and the in-plane y axis uses the row spacing.
        spacing_y, spacing_x = map(float, dcm.PixelSpacing)
        pos_x = float(dcm.ImagePositionPatient[0])
        pos_y = float(dcm.ImagePositionPatient[2])

        # 1. Create standardized physical coordinates (Y-axis increasing upwards)
        x_coords = (np.arange(width) * spacing_x) + pos_x
        y_coords = (np.arange(height) * spacing_y) + pos_y

        # Data grid needs to be flipped vertically to match the coordinate system
        data_grid = np.flipud(pixel_data)

        # Y-coordinates are now monotonically increasing after flipping the data
        # No need to reverse y_coords as we're using the original coordinates


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
            "pixel_spacing_x": spacing_x,
            "pixel_spacing_y": spacing_y,
        }

        # 4. Create and return the standard data object
        return StandardDoseData(data_grid, x_coords, y_coords, metadata)

    except Exception as e:
        logger.error(f"Failed to load DICOM file {filename}: {e}", exc_info=True)
        raise
