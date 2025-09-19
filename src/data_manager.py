from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from src.standard_data_model import StandardDoseData, ROI_Data

class DataManager:
    """
    A dedicated class to manage all application data.
    This includes loaded files, analysis results, and intermediate data.
    """
    def __init__(self):
        # Loaded data
        self.dicom_data: Optional[StandardDoseData] = None
        self.mcc_data: Optional[StandardDoseData] = None

        # ROI (Region of Interest) data
        self.dicom_roi: Optional[ROI_Data] = None
        self.mcc_roi: Optional[ROI_Data] = None

        # DICOM origin and coordinate data
        self.initial_dicom_phys_coords: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.initial_dicom_pixel_origin: Optional[Tuple[int, int]] = None

        # Profile-related data
        self.profile_line: Optional[Dict[str, Any]] = None
        self.current_profile_data: Optional[Dict[str, Any]] = None

        # Gamma analysis results
        self.gamma_map: Optional[np.ndarray] = None
        self.gamma_stats: Optional[Dict[str, float]] = None
        self.phys_extent: Optional[List[float]] = None
        self.mcc_interp_data: Optional[np.ndarray] = None

        # DD and DTA results
        self.dd_map: Optional[np.ndarray] = None
        self.dta_map: Optional[np.ndarray] = None
        self.dd_stats: Optional[Dict[str, float]] = None
        self.dta_stats: Optional[Dict[str, float]] = None
