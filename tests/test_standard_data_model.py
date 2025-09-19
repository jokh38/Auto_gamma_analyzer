import unittest
import numpy as np

# 이 테스트는 초기에 실패해야 합니다. ROI_Data가 아직 정의되지 않았기 때문입니다.
from src.standard_data_model import ROI_Data, StandardDoseData

class TestROI_Data(unittest.TestCase):

    def test_roi_data_creation(self):
        """
        ROI_Data 객체가 명세서에 따라 올바르게 생성되는지 테스트합니다.
        """
        # given: ROI 데이터를 위한 샘플 데이터
        dose_grid = np.random.rand(10, 10)
        x_coords = np.arange(10)
        y_coords = np.arange(10)
        x_indices = np.arange(10)
        y_indices = np.arange(10)
        physical_extent = [0.0, 9.0, 0.0, 9.0]
        source_metadata = {'patient_name': 'Test Patient'}

        # when: ROI_Data 객체 생성
        roi = ROI_Data(
            dose_grid=dose_grid,
            x_coords=x_coords,
            y_coords=y_coords,
            x_indices=x_indices,
            y_indices=y_indices,
            physical_extent=physical_extent,
            source_metadata=source_metadata
        )

        # then: 속성들이 올바르게 할당되었는지 확인
        np.testing.assert_array_equal(roi.dose_grid, dose_grid)
        np.testing.assert_array_equal(roi.x_coords, x_coords)
        np.testing.assert_array_equal(roi.y_coords, y_coords)
        np.testing.assert_array_equal(roi.x_indices, x_indices)
        np.testing.assert_array_equal(roi.y_indices, y_indices)
        self.assertEqual(roi.physical_extent, physical_extent)
        self.assertEqual(roi.source_metadata, source_metadata)

if __name__ == '__main__':
    unittest.main()
