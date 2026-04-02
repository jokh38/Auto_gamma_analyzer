import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis import extract_profile_data, perform_gamma_analysis


class DenseHandler:
    def __init__(self, data, x_coords, y_coords):
        self.pixel_data = np.array(data, dtype=float)
        self.normalization_factor = 1.0
        self.phys_x_mesh, self.phys_y_mesh = np.meshgrid(x_coords, y_coords)
        self.physical_extent = [float(np.min(x_coords)), float(np.max(x_coords)), float(np.min(y_coords)), float(np.max(y_coords))]
        self.filename = "dense.dcm"
        self.crop_pixel_offset = (0, 0)

    def get_pixel_data(self):
        return self.pixel_data * self.normalization_factor

    def set_normalization_factor(self, factor):
        self.normalization_factor = float(factor)

    def get_physical_extent(self):
        return self.physical_extent


class SparseHandler(DenseHandler):
    def __init__(self, data, x_coords, y_coords):
        super().__init__(data, x_coords, y_coords)
        self.matrix_data = self.pixel_data.copy()
        self.filename = "sparse.mcc"

    def get_matrix_data(self):
        return self.matrix_data * self.normalization_factor


class TestAnalysisFunctions(unittest.TestCase):
    def test_extract_profile_data_with_dense_file_b(self):
        x_coords = np.array([0.0, 10.0, 20.0])
        y_coords = np.array([20.0, 10.0, 0.0])

        file_a = DenseHandler(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            x_coords=x_coords,
            y_coords=y_coords,
        )
        file_b = DenseHandler(
            data=[[2, 3, 4], [5, 6, 7], [8, 9, 10]],
            x_coords=x_coords,
            y_coords=y_coords,
        )

        profile = extract_profile_data(
            direction="vertical",
            fixed_position=10.0,
            dicom_handler=file_a,
            mcc_handler=file_b,
        )

        np.testing.assert_array_equal(profile["phys_coords"], np.array([0.0, 10.0, 20.0]))
        np.testing.assert_array_equal(profile["dicom_values"], np.array([8.0, 5.0, 2.0]))
        np.testing.assert_array_equal(profile["mcc_phys_coords"], np.array([0.0, 10.0, 20.0]))
        np.testing.assert_array_equal(profile["mcc_values"], np.array([9.0, 6.0, 3.0]))
        np.testing.assert_array_equal(profile["dicom_at_mcc"], np.array([8.0, 5.0, 2.0]))

    def test_extract_profile_data_with_sparse_file_b(self):
        x_coords = np.array([0.0, 10.0, 20.0])
        y_coords = np.array([20.0, 10.0, 0.0])

        file_a = DenseHandler(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            x_coords=x_coords,
            y_coords=y_coords,
        )
        file_b = SparseHandler(
            data=[[2, -1, 4], [5, 6, -1], [8, 9, 10]],
            x_coords=x_coords,
            y_coords=y_coords,
        )

        profile = extract_profile_data(
            direction="horizontal",
            fixed_position=10.0,
            dicom_handler=file_a,
            mcc_handler=file_b,
        )

        np.testing.assert_array_equal(profile["phys_coords"], np.array([0.0, 10.0, 20.0]))
        np.testing.assert_array_equal(profile["dicom_values"], np.array([4.0, 5.0, 6.0]))
        np.testing.assert_array_equal(profile["mcc_phys_coords"], np.array([0.0, 10.0]))
        np.testing.assert_array_equal(profile["mcc_values"], np.array([5.0, 6.0]))
        np.testing.assert_array_equal(profile["dicom_at_mcc"], np.array([4.0, 5.0]))

    def test_perform_gamma_analysis_with_dense_reference(self):
        coords = np.arange(5, dtype=float)
        ref_data = np.zeros((5, 5), dtype=float)
        eval_data = np.zeros((5, 5), dtype=float)

        ref_data[1:4, 1:4] = 100.0
        eval_data[1:4, 2:5] = 100.0

        reference_handler = DenseHandler(ref_data, coords, coords)
        evaluation_handler = DenseHandler(eval_data, coords, coords)

        (
            gamma_map,
            stats,
            _,
            interp_ref,
            dd_map,
            dta_map,
            dd_stats,
            dta_stats,
            gamma_map_interp,
            _,
            _,
        ) = perform_gamma_analysis(
            reference_handler=reference_handler,
            evaluation_handler=evaluation_handler,
            dose_percent_threshold=3,
            distance_mm_threshold=3,
        )

        self.assertEqual(gamma_map.shape, ref_data.shape)
        self.assertEqual(interp_ref.shape, eval_data.shape)
        self.assertEqual(dd_map.shape, ref_data.shape)
        self.assertEqual(dta_map.shape, ref_data.shape)
        self.assertEqual(gamma_map_interp.shape, eval_data.shape)
        self.assertIn("pass_rate", stats)
        self.assertIn("total_reference_points", stats)
        self.assertIn("evaluated_points", stats)
        self.assertIn("passed_points", stats)
        self.assertIn("failed_points", stats)
        self.assertEqual(stats["total_reference_points"], 25)
        self.assertEqual(stats["evaluated_points"], stats["total_points"])
        self.assertEqual(stats["passed_points"] + stats["failed_points"], stats["evaluated_points"])
        self.assertGreater(stats["total_points"], 0)
        self.assertLessEqual(stats["pass_rate"], 100.0)
        self.assertGreater(stats["mean"], 0.0)
        self.assertGreater(dd_stats["total_points"], 0)
        self.assertGreater(dta_stats["total_points"], 0)

    def test_extract_profile_data_applies_normalization_factors(self):
        x_coords = np.array([0.0, 10.0, 20.0])
        y_coords = np.array([20.0, 10.0, 0.0])

        file_a = DenseHandler(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            x_coords=x_coords,
            y_coords=y_coords,
        )
        file_b = SparseHandler(
            data=[[2, -1, 4], [5, 6, -1], [8, 9, 10]],
            x_coords=x_coords,
            y_coords=y_coords,
        )
        file_a.set_normalization_factor(1.1)
        file_b.set_normalization_factor(1.7)

        profile = extract_profile_data(
            direction="horizontal",
            fixed_position=10.0,
            dicom_handler=file_a,
            mcc_handler=file_b,
        )

        np.testing.assert_allclose(profile["dicom_values"], np.array([4.4, 5.5, 6.6]))
        np.testing.assert_allclose(profile["mcc_values"], np.array([8.5, 10.2]))
        np.testing.assert_allclose(profile["dicom_at_mcc"], np.array([4.4, 5.5]))


if __name__ == "__main__":
    unittest.main()
