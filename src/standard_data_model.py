import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class StandardDoseData:
    """
    A standardized data object for holding 2D dose data.

    This class ensures that all data loaded from various file formats (DICOM, MCC, etc.)
    is represented in a consistent, easy-to-use structure for analysis and visualization.

    Attributes:
        data_grid (np.ndarray): A 2D numpy array representing the dose grid.
                                This grid is always dense (interpolated if necessary).
        x_coords (np.ndarray): A 1D numpy array of the physical X-coordinates (in mm).
                               Guaranteed to be monotonically increasing.
        y_coords (np.ndarray): A 1D numpy array of the physical Y-coordinates (in mm).
                               Guaranteed to be monotonically increasing.
        metadata (Dict[str, Any]): A dictionary containing additional information
                                   about the data, such as patient info, device type, etc.
    """
    data_grid: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Perform validation checks after initialization."""
        if self.data_grid.ndim != 2:
            raise ValueError(f"data_grid must be a 2D array, but got {self.data_grid.ndim} dimensions.")

        if self.x_coords.ndim != 1 or self.y_coords.ndim != 1:
            raise ValueError("x_coords and y_coords must be 1D arrays.")

        if self.data_grid.shape[1] != len(self.x_coords):
            raise ValueError(f"Width of data_grid ({self.data_grid.shape[1]}) does not match length of x_coords ({len(self.x_coords)}).")

        if self.data_grid.shape[0] != len(self.y_coords):
            raise ValueError(f"Height of data_grid ({self.data_grid.shape[0]}) does not match length of y_coords ({len(self.y_coords)}).")

        if np.any(np.diff(self.x_coords) <= 0):
            raise ValueError("x_coords must be monotonically increasing.")

        if np.any(np.diff(self.y_coords) <= 0):
            raise ValueError("y_coords must be monotonically increasing.")

    @property
    def physical_extent(self) -> list[float]:
        """
        Returns the physical coordinate range [xmin, xmax, ymin, ymax],
        compatible with matplotlib's extent parameter.
        """
        # To calculate the extent for imshow, we need to consider the edges of the pixels.
        # We assume the coordinates represent the center of the pixels.
        dx = (self.x_coords[1] - self.x_coords[0]) / 2.0 if len(self.x_coords) > 1 else 0.5
        dy = (self.y_coords[1] - self.y_coords[0]) / 2.0 if len(self.y_coords) > 1 else 0.5

        return [
            self.x_coords[0] - dx,
            self.x_coords[-1] + dx,
            self.y_coords[0] - dy,
            self.y_coords[-1] + dy
        ]

@dataclass
class ROI_Data:
    """
    Holds data for a specific Region of Interest (ROI).

    This object contains not only the cropped dose grid but also the
    corresponding coordinates and indices from the original data,
    facilitating analysis without repeated calculations.

    Attributes:
        dose_grid (np.ndarray): 2D array of dose values within the ROI.
        x_coords (np.ndarray): 1D array of physical X-coordinates for the ROI.
        y_coords (np.ndarray): 1D array of physical Y-coordinates for the ROI.
        x_indices (np.ndarray): 1D array of original column indices for the ROI.
        y_indices (np.ndarray): 1D array of original row indices for the ROI.
        physical_extent (list[float]): Physical boundaries [xmin, xmax, ymin, ymax].
        source_metadata (dict): Metadata from the source StandardDoseData.
    """
    dose_grid: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    x_indices: np.ndarray
    y_indices: np.ndarray
    physical_extent: list[float]
    source_metadata: dict
