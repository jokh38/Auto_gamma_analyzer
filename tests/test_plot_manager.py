import unittest
import os
import sys
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_manager import DataManager
from src.load_dcm import load_dcm
from src.load_mcc import load_mcc
from src.analysis import extract_profile_data
from src.standard_data_model import StandardDoseData
from src.ui_components import PlotManager
from src.app_controller import AppController # Need this to create ROI
from unittest.mock import Mock

# The temporary PlotManager is no longer needed. We test the real one.

class TestPlotManager(unittest.TestCase):
    """Tests for the PlotManager class."""

    def setUp(self):
        """Set up a DataManager with loaded data and ROIs for each test."""
        self.data_manager = DataManager()
        dicom_path = os.path.join('example_data', '1G240_2cm.dcm')
        mcc_path = os.path.join('example_data', '1G240_2cm.mcc')
        self.data_manager.dicom_data = load_dcm(dicom_path)
        self.data_manager.mcc_data = load_mcc(mcc_path)

        # Create an AppController instance just to use its _extract_roi_from_data method
        controller = AppController(Mock(), self.data_manager, Mock())
        self.data_manager.dicom_roi = controller._extract_roi_from_data(self.data_manager.dicom_data)

        # Set a default vertical profile line for testing
        self.data_manager.profile_line = {"type": "vertical", "x": 0}

    def test_generate_profile_data_preparation(self):
        """
        Tests that PlotManager correctly prepares the data needed for plotting a profile using handlers.
        """
        # 1. Create handlers for DICOM and MCC data
        from src.file_handlers import DicomFileHandler, MCCFileHandler

        dicom_handler = DicomFileHandler()
        dicom_handler.open_file('example_data/1G240_2cm.dcm')
        self.data_manager.dicom_handler = dicom_handler

        mcc_handler = MCCFileHandler()
        mcc_handler.open_file('example_data/1G240_2cm.mcc')
        # Crop MCC to match DICOM bounds
        if dicom_handler.dose_bounds:
            mcc_handler.crop_to_bounds(dicom_handler.dose_bounds)
        self.data_manager.mcc_handler = mcc_handler

        # 2. Create PlotManager with the pre-loaded DataManager.
        #    We pass None for the UI components as they are not needed for this test.
        plot_manager = PlotManager(self.data_manager, None, None, None, None, None)

        # 3. Call the method to generate profile data
        profile_data = plot_manager.generate_profile_data()

        # 4. Assert that the returned data is valid and well-formed
        self.assertIsNotNone(profile_data, "Profile data dictionary should not be None.")

        # Check for the presence of essential keys
        expected_keys = ['phys_coords', 'dicom_values']
        for key in expected_keys:
            self.assertIn(key, profile_data, f"Key '{key}' should be in profile_data.")

        # Check that data has the expected shape/size (i.e., not empty)
        self.assertIsInstance(profile_data['phys_coords'], np.ndarray)
        self.assertGreater(len(profile_data['phys_coords']), 0)
        self.assertEqual(len(profile_data['phys_coords']), len(profile_data['dicom_values']),
                         "Physical coordinates and DICOM values should have the same length.")

        print("PlotManager Test: PASS")

    def test_get_profile_gamma_values_for_vertical_profile(self):
        class ReferenceHandler:
            def __init__(self):
                x_coords = np.array([0.0, 10.0, 20.0])
                y_coords = np.array([20.0, 10.0, 0.0])
                self.phys_x_mesh, self.phys_y_mesh = np.meshgrid(x_coords, y_coords)

        self.data_manager.file_b_handler = ReferenceHandler()
        self.data_manager.gamma_map = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])

        plot_manager = PlotManager(self.data_manager, None, None, None, None, None)
        profile_data = {
            "type": "vertical",
            "fixed_pos": 10.0,
            "mcc_phys_coords": np.array([20.0, 10.0, 0.0]),
        }

        gamma_values = plot_manager._get_profile_gamma_values(profile_data)

        np.testing.assert_allclose(gamma_values, np.array([0.2, 0.5, 0.8]))

if __name__ == '__main__':
    unittest.main()
