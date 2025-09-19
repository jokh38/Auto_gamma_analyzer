import unittest
import os
import sys
from typing import Optional

# Add the project root to the Python path to allow for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.standard_data_model import StandardDoseData
from src.load_dcm import load_dcm
from src.data_manager import DataManager

from src.standard_data_model import ROI_Data

class TestDataManager(unittest.TestCase):
    """Tests for the DataManager class."""

    def setUp(self):
        """Set up a new DataManager instance for each test."""
        self.data_manager = DataManager()

    def test_initial_attributes(self):
        """
        Tests that the DataManager is initialized with the correct attributes
        after the refactoring.
        """
        # Check that new ROI attributes exist and are None
        self.assertTrue(hasattr(self.data_manager, 'dicom_roi'), "DataManager should have 'dicom_roi' attribute")
        self.assertIsNone(self.data_manager.dicom_roi, "dicom_roi should be initialized to None")

        self.assertTrue(hasattr(self.data_manager, 'mcc_roi'), "DataManager should have 'mcc_roi' attribute")
        self.assertIsNone(self.data_manager.mcc_roi, "mcc_roi should be initialized to None")

        # Check that the old dose_bounds attribute is gone
        self.assertFalse(hasattr(self.data_manager, 'dose_bounds'), "DataManager should no longer have 'dose_bounds' attribute")

    def test_dicom_data_assignment(self):
        """
        Tests that DICOM data can be loaded and assigned to the DataManager.
        This confirms the basic mechanism works before refactoring.
        """
        # The path is relative to the project root, where tests are expected to run
        dicom_file_path = os.path.join('example_data', '1G240_2cm.dcm')
        loaded_data = load_dcm(dicom_file_path)

        # Assign the loaded data object to the manager's attribute
        self.data_manager.dicom_data = loaded_data

        # Assert that the data is present and of the correct type
        self.assertIsNotNone(self.data_manager.dicom_data)
        self.assertIsInstance(self.data_manager.dicom_data, StandardDoseData)

if __name__ == '__main__':
    unittest.main()
