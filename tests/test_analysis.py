import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.standard_data_model import ROI_Data
from src.analysis import extract_profile_data, perform_gamma_analysis

class TestAnalysisFunctions(unittest.TestCase):

    def test_extract_profile_data_with_roi(self):
        """
        Tests the refactored extract_profile_data function with an ROI_Data object.
        """
        # 1. Given: Create a sample ROI_Data object
        roi = ROI_Data(
            dose_grid=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            x_coords=np.array([10, 20, 30]),
            y_coords=np.array([5, 15, 25]),
            x_indices=np.arange(3),
            y_indices=np.arange(3),
            physical_extent=[5, 35, 0, 30],
            source_metadata={}
        )

        # 2. When: Extract a vertical profile at x=20 (the middle column)
        profile = extract_profile_data(direction="vertical", fixed_position=20, roi_data=roi)

        # 3. Then: Assert the profile is correct
        self.assertIsNotNone(profile)
        np.testing.assert_array_equal(profile['phys_coords'], roi.y_coords)
        np.testing.assert_array_equal(profile['dicom_values'], [2, 5, 8]) # Middle column
        self.assertEqual(profile['type'], "vertical")
        self.assertEqual(profile['fixed_pos'], 20)

        # When: Extract a horizontal profile at y=15 (the middle row)
        profile = extract_profile_data(direction="horizontal", fixed_position=15, roi_data=roi)

        # Then: Assert the profile is correct
        self.assertIsNotNone(profile)
        np.testing.assert_array_equal(profile['phys_coords'], roi.x_coords)
        np.testing.assert_array_equal(profile['dicom_values'], [4, 5, 6]) # Middle row
        self.assertEqual(profile['type'], "horizontal")
        self.assertEqual(profile['fixed_pos'], 15)


    def test_perform_gamma_analysis_with_roi(self):
        """
        Tests the refactored perform_gamma_analysis function with ROI_Data objects.
        """
        # 1. Given: Create two simple ROI_Data objects that should be easy to compare.
        # Let's make the eval data a slightly shifted version of the ref data.
        ref_grid = np.zeros((10, 10))
        ref_grid[3:7, 3:7] = 100 # A hot spot

        eval_grid = np.zeros((10, 10))
        eval_grid[3:7, 4:8] = 105 # Shifted right and slightly higher dose

        common_coords = np.arange(10)

        # Create a 4x4 grid of reference points to satisfy the cubic interpolator
        xs = np.linspace(4, 6, 4)
        ys = np.linspace(4, 6, 4)
        ref_points_x, ref_points_y = np.meshgrid(xs, ys)
        ref_coords = np.vstack([ref_points_x.ravel(), ref_points_y.ravel()]).T
        ref_values = np.full(ref_coords.shape[0], 100)

        ref_roi = ROI_Data(
            dose_grid=ref_grid, x_coords=common_coords, y_coords=common_coords,
            x_indices=common_coords, y_indices=common_coords,
            physical_extent=[0,9,0,9], source_metadata={'original_points': {'coords': ref_coords, 'values': ref_values}}
        )

        eval_roi = ROI_Data(
            dose_grid=eval_grid, x_coords=common_coords, y_coords=common_coords,
            x_indices=common_coords, y_indices=common_coords,
            physical_extent=[0,9,0,9], source_metadata={}
        )

        # 2. When: Perform gamma analysis.
        # The core change is passing ROI objects directly. The threshold parameter is removed.
        gamma_map, stats, _, _, _, _, _, _ = perform_gamma_analysis(
            reference_roi=ref_roi,
            evaluation_roi=eval_roi,
            dose_percent_threshold=3,
            distance_mm_threshold=3
        )

        # 3. Then: Assert that the results are reasonable.
        # A detailed check is complex, so we'll check the pass rate.
        # Given the shift and dose difference, the pass rate should not be 100%.
        self.assertIsNotNone(stats)
        self.assertIn('pass_rate', stats)
        self.assertLess(stats['pass_rate'], 100)

if __name__ == '__main__':
    unittest.main()
