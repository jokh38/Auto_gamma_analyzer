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
from unittest.mock import Mock, MagicMock, patch
from src.app_controller import AppController
from src.standard_data_model import StandardDoseData, ROI_Data

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

        # Use a real PlotManager, but with mocked canvas dependencies
        mock_canvas = MagicMock()
        mock_table = MagicMock()
        self.plot_manager = PlotManager(
            data_manager=self.data_manager,
            dicom_canvas=mock_canvas,
            mcc_canvas=mock_canvas,
            profile_canvas=mock_canvas,
            gamma_canvas=mock_canvas,
            profile_table=mock_table
        )

        # Mock the main view (QMainWindow) and its UI widgets.
        # This allows us to test the controller's logic without a live GUI.
        self.mock_view = Mock()
        self.mock_view.dta_spin.value.return_value = 3
        self.mock_view.dd_spin.value.return_value = 3
        self.mock_view.gamma_type_combo.currentText.return_value = "Global"

        # Controller instance for tests
        self.controller = AppController(self.mock_view, self.data_manager, self.plot_manager)


    def test_extract_roi_from_data(self):
        """
        Tests the logic for extracting an ROI from a StandardDoseData object.
        This test replaces the need to test the old _calculate_dose_bounds.
        """
        # 1. Given: Create a sample StandardDoseData object with a predictable dose pattern
        x_coords = np.linspace(-50, 50, 101)  # 101 points from -50 to 50
        y_coords = np.linspace(-50, 50, 101)
        # Create a grid with a "hot spot" in the middle
        data_grid = np.zeros((101, 101))
        data_grid[40:61, 40:61] = 100  # A 21x21 square of 100 in the center

        sample_data = StandardDoseData(
            data_grid=data_grid,
            x_coords=x_coords,
            y_coords=y_coords,
            metadata={'source': 'test'}
        )

        # 2. When: Call the function to extract the ROI (using a 10% threshold)
        # The function is private, so we access it with a leading underscore
        roi_result = self.controller._extract_roi_from_data(sample_data, threshold_percent=10)

        # 3. Then: Assert the result is a valid ROI_Data object with correct values
        self.assertIsInstance(roi_result, ROI_Data)

        # The hot spot is from index 40 to 60.
        # Check that the indices are correct.
        np.testing.assert_array_equal(roi_result.y_indices, np.arange(40, 61))
        np.testing.assert_array_equal(roi_result.x_indices, np.arange(40, 61))

        # Check that the data grid is the 21x21 square
        self.assertEqual(roi_result.dose_grid.shape, (21, 21))
        np.testing.assert_allclose(roi_result.dose_grid, 100)

        # Check that the coordinates match the indices
        np.testing.assert_allclose(roi_result.x_coords, x_coords[40:61])
        np.testing.assert_allclose(roi_result.y_coords, y_coords[40:61])

        # Check metadata was passed through
        self.assertEqual(roi_result.source_metadata, {'source': 'test'})


    @patch('src.app_controller.load_dcm')
    @patch('src.app_controller.load_mcc')
    @patch('src.app_controller.QFileDialog.getOpenFileName')
    def test_load_files_creates_roi(self, mock_get_open_filename, mock_load_mcc, mock_load_dcm):
        """
        Tests that loading files correctly calls the ROI extraction and populates the DataManager.
        """
        # 1. Given: Mock the file dialog and data loaders
        # Mock file dialog to return a dummy path
        mock_get_open_filename.return_value = ('dummy_path.dcm', '')

        # Mock loaders to return a more realistic StandardDoseData object
        mock_dicom_data = MagicMock(spec=StandardDoseData)
        mock_dicom_data.x_coords = np.array([])
        mock_dicom_data.y_coords = np.array([])
        mock_dicom_data.metadata = {'image_position_patient': [0,0,0], 'pixel_spacing': [1,1]}
        mock_load_dcm.return_value = mock_dicom_data

        # Mock the ROI extraction function to track its calls and return a specific ROI_Data object
        mock_roi_data = MagicMock(spec=ROI_Data)
        mock_roi_data.dose_grid = np.array([[1]])
        mock_roi_data.physical_extent = [0,1,0,1]
        mock_roi_data.x_coords = np.array([0])
        mock_roi_data.y_coords = np.array([0])
        # Add source_metadata to prevent AttributeError in the modified generate_profile_data
        mock_roi_data.source_metadata = {}
        self.controller._extract_roi_from_data = MagicMock(return_value=mock_roi_data)

        # 2. When: Call the load_dicom_file method
        self.controller.load_dicom_file()

        # 3. Then: Assert that ROI extraction was called and the result was stored
        self.controller._extract_roi_from_data.assert_called_once_with(mock_dicom_data)
        self.assertIs(self.data_manager.dicom_roi, mock_roi_data)

        # --- Test MCC loading ---
        # Reset mocks for the next part of the test
        self.controller._extract_roi_from_data.reset_mock()
        mock_get_open_filename.return_value = ('dummy_path.mcc', '')
        mock_mcc_data = MagicMock(spec=StandardDoseData)
        mock_mcc_data.metadata = {'device': 'test_device'}
        mock_load_mcc.return_value = mock_mcc_data

        # When: Call the load_measurement_file method
        self.controller.load_measurement_file()

        # Then: Assert that ROI extraction was called and the result was stored
        self.controller._extract_roi_from_data.assert_called_once_with(mock_mcc_data)
        self.assertIs(self.data_manager.mcc_roi, mock_roi_data)


    def test_run_gamma_analysis_logic(self):
        """
        Tests that AppController correctly runs gamma analysis and populates the DataManager.
        """
        # Given: Manually create ROIs since setUp no longer does it.
        self.data_manager.dicom_roi = self.controller._extract_roi_from_data(self.data_manager.dicom_data)
        self.data_manager.mcc_roi = self.controller._extract_roi_from_data(self.data_manager.mcc_data)

        # Check pre-conditions
        self.assertIsNone(self.data_manager.gamma_stats, "gamma_stats should be None before analysis.")

        # 2. Run the analysis
        self.controller.run_gamma_analysis()

        # 4. Assert that the analysis populated the data manager
        self.assertIsNotNone(self.data_manager.gamma_stats, "gamma_stats should be populated after analysis.")

        # Check for expected keys and plausible values in the results
        self.assertIn('pass_rate', self.data_manager.gamma_stats)
        self.assertIsInstance(self.data_manager.gamma_stats['pass_rate'], float)
        self.assertGreaterEqual(self.data_manager.gamma_stats['pass_rate'], 0.0)

        self.assertIsNotNone(self.data_manager.gamma_map, "gamma_map should be populated.")
        self.assertIsInstance(self.data_manager.gamma_map, np.ndarray)

    @patch('src.ui_components.draw_image')
    @patch('src.app_controller.load_dcm')
    @patch('src.app_controller.QFileDialog.getOpenFileName')
    def test_redraw_all_uses_roi_data(self, mock_get_open_filename, mock_load_dcm, mock_draw_image):
        """
        Tests that the image drawing function is called with ROI data, not full data.
        """
        # 1. Given: Mock file loading to return data and trigger ROI creation
        mock_get_open_filename.return_value = ('dummy_path.dcm', '')

        mock_dicom_data = StandardDoseData(
            data_grid=np.ones((100, 100)),
            x_coords=np.arange(100),
            y_coords=np.arange(100),
            metadata={'image_position_patient': [0,0,0], 'pixel_spacing': [1,1]}
        )
        mock_load_dcm.return_value = mock_dicom_data

        # Create a specific ROI object that we expect to be used
        mock_roi = ROI_Data(
            dose_grid=np.ones((10, 10)),
            x_coords=np.arange(10),
            y_coords=np.arange(10),
            x_indices=np.arange(10),
            y_indices=np.arange(10),
            physical_extent=[0, 9, 0, 9],
            source_metadata={}
        )
        # Mock the ROI extraction function to return our specific ROI. Make it more complete
        # to satisfy the profile generation that is also triggered.
        mock_roi.x_coords = np.arange(10)
        mock_roi.y_coords = np.arange(10)
        self.controller._extract_roi_from_data = MagicMock(return_value=mock_roi)

        # 2. When: Load a file, which should trigger a redraw
        self.controller.load_dicom_file()

        # 3. Then: Assert that draw_image was called with the ROI's data
        # We expect it to be called for the DICOM canvas.
        mock_draw_image.assert_called()
        call_args, call_kwargs = mock_draw_image.call_args

        # Check positional/keyword arguments passed to draw_image
        np.testing.assert_array_equal(call_kwargs['image_data'], mock_roi.dose_grid)
        self.assertEqual(call_kwargs['extent'], mock_roi.physical_extent)

        # Check that obsolete arguments are no longer passed
        self.assertNotIn('apply_cropping', call_kwargs)
        self.assertNotIn('crop_bounds', call_kwargs)

if __name__ == '__main__':
    unittest.main()
