import unittest
import os
import sys
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_manager import DataManager
from src.ui_components import PlotManager
from src.load_dcm import load_dcm
from src.load_mcc import load_mcc
from unittest.mock import Mock
from src.app_controller import AppController

# The temporary AppController is no longer needed.

class TestAppController(unittest.TestCase):
    """Tests for the AppController class."""

    def setUp(self):
        """Set up with loaded data for each test."""
        self.data_manager = DataManager()
        dicom_path = os.path.join('example_data', '1G240_2cm.dcm')
        mcc_path = os.path.join('example_data', '1G240_2cm.mcc')
        self.data_manager.dicom_data = load_dcm(dicom_path)
        self.data_manager.mcc_data = load_mcc(mcc_path)

        # Mock the PlotManager, as its functionality is not under test.
        # We only need to ensure its methods can be called without error.
        self.mock_plot_manager = Mock(spec=PlotManager)

        # Mock the main view (QMainWindow) and its UI widgets.
        # This allows us to test the controller's logic without a live GUI.
        self.mock_view = Mock()
        self.mock_view.dta_spin.value.return_value = 3
        self.mock_view.dd_spin.value.return_value = 3
        self.mock_view.gamma_type_combo.currentText.return_value = "Global"

    def test_run_gamma_analysis_logic(self):
        """
        Tests that AppController correctly runs gamma analysis and populates the DataManager.
        """
        # 1. Create an instance of the real controller with mocked dependencies
        controller = AppController(self.mock_view, self.data_manager, self.mock_plot_manager)

        # 2. Check pre-conditions
        self.assertIsNone(self.data_manager.gamma_stats, "gamma_stats should be None before analysis.")

        # 3. Run the analysis
        controller.run_gamma_analysis()

        # 4. Assert that the analysis populated the data manager
        self.assertIsNotNone(self.data_manager.gamma_stats, "gamma_stats should be populated after analysis.")

        # Check for expected keys and plausible values in the results
        self.assertIn('pass_rate', self.data_manager.gamma_stats)
        self.assertIsInstance(self.data_manager.gamma_stats['pass_rate'], float)
        self.assertGreaterEqual(self.data_manager.gamma_stats['pass_rate'], 0.0)

        self.assertIsNotNone(self.data_manager.gamma_map, "gamma_map should be populated.")
        self.assertIsInstance(self.data_manager.gamma_map, np.ndarray)

        print("AppController Test: PASS")

if __name__ == '__main__':
    unittest.main()
