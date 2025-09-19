import unittest
import os
import sys
from typing import Optional

# Add the project root to the Python path to allow for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.standard_data_model import StandardDoseData
from src.load_dcm import load_dcm
from src.data_manager import DataManager

# The temporary DataManager class is no longer needed, as we now import the real one.

class TestDataManager(unittest.TestCase):
    """Tests for the DataManager class."""

    def test_dicom_data_assignment(self):
        """
        Tests that DICOM data can be loaded and assigned to the DataManager.
        This confirms the basic mechanism works before refactoring.
        """
        # 1. Create an instance of the temporary DataManager
        data_manager = DataManager()

        # 2. Load data using the existing loader function
        # The path is relative to the project root, where tests are expected to run
        dicom_file_path = os.path.join('example_data', '1G240_2cm.dcm')
        loaded_data = load_dcm(dicom_file_path)

        # 3. Assign the loaded data object to the manager's attribute
        data_manager.dicom_data = loaded_data

        # 4. Assert that the data is present and of the correct type
        self.assertIsNotNone(data_manager.dicom_data)
        self.assertIsInstance(data_manager.dicom_data, StandardDoseData)

        # As requested by the user's plan, print a success message
        print("DataManager Test: PASS")

if __name__ == '__main__':
    unittest.main()
