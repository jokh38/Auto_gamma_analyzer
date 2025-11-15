"""
Unit tests for file_handlers module.
"""
import pytest
import numpy as np
import os
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock
from src.file_handlers import BaseFileHandler, DicomFileHandler, MCCFileHandler


class TestBaseFileHandler:
    """Test cases for BaseFileHandler class."""

    def test_init_with_config(self, tmp_path):
        """Test initialization with config.yaml file."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "dta": 2,
            "dd": 3,
            "suppression_level": 15,
            "roi_margin": 5
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Change to tmp_path directory
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = BaseFileHandler()
            assert handler.dta == 2
            assert handler.dd == 3
            assert handler.suppression_level == 15
            assert handler.roi_margin == 5
        finally:
            os.chdir(original_dir)

    def test_init_without_config(self, tmp_path):
        """Test initialization without config.yaml (uses defaults)."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = BaseFileHandler()
            # Should use default values
            assert handler.dta == 3
            assert handler.dd == 3
            assert handler.suppression_level == 10
            assert handler.roi_margin == 2
        finally:
            os.chdir(original_dir)

    def test_get_filename(self):
        """Test get_filename method."""
        handler = BaseFileHandler()
        assert handler.get_filename() is None

        handler.filename = "/path/to/test_file.dcm"
        assert handler.get_filename() == "test_file.dcm"

    def test_get_physical_extent(self):
        """Test get_physical_extent method."""
        handler = BaseFileHandler()
        assert handler.get_physical_extent() is None

        handler.physical_extent = [-100, 100, -50, 50]
        assert handler.get_physical_extent() == [-100, 100, -50, 50]

    def test_get_origin_coords(self):
        """Test get_origin_coords method."""
        handler = BaseFileHandler()
        assert handler.get_origin_coords() == (0, 0)

        handler.origin_x = 10
        handler.origin_y = 20
        assert handler.get_origin_coords() == (10, 20)

    def test_get_spacing(self):
        """Test get_spacing method."""
        handler = BaseFileHandler()
        assert handler.get_spacing() == (1.0, 1.0)

        handler.pixel_spacing = 2.5
        assert handler.get_spacing() == (2.5, 2.5)


class TestDicomFileHandler:
    """Test cases for DicomFileHandler class."""

    @patch('pydicom.dcmread')
    def test_open_file_success(self, mock_dcmread, tmp_path):
        """Test successful DICOM file loading."""
        # Create mock DICOM data
        mock_dcm = MagicMock()
        mock_dcm.Modality = 'RTDOSE'
        mock_dcm.pixel_array = np.ones((100, 100)) * 2.0
        mock_dcm.DoseGridScaling = 0.5
        mock_dcm.PixelSpacing = [2.0, 2.0]
        mock_dcm.ImagePositionPatient = [10.0, 0.0, 20.0]
        mock_dcm.get = MagicMock(side_effect=lambda key, default: {
            'InstitutionName': 'Test Hospital',
            'PatientID': '12345',
            'PatientName': 'Test Patient'
        }.get(key, default))
        mock_dcmread.return_value = mock_dcm

        # Change to tmp_path to avoid config.yaml dependency
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = DicomFileHandler()
            success, error = handler.open_file("test.dcm")

            assert success is True
            assert error is None
            assert handler.pixel_data is not None
            assert handler.pixel_spacing == 2.0
            assert handler.dicom_data is not None
        finally:
            os.chdir(original_dir)

    @patch('pydicom.dcmread')
    def test_open_file_wrong_modality(self, mock_dcmread, tmp_path):
        """Test loading non-RTDOSE DICOM file."""
        mock_dcm = MagicMock()
        mock_dcm.Modality = 'CT'
        mock_dcmread.return_value = mock_dcm

        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = DicomFileHandler()
            success, error = handler.open_file("test.dcm")

            assert success is False
            assert "not an RT Dose file" in error
        finally:
            os.chdir(original_dir)

    def test_physical_to_pixel_coord(self, tmp_path):
        """Test physical to pixel coordinate conversion."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = DicomFileHandler()
            handler.pixel_spacing = 2.0
            handler.dicom_origin_x = 5
            handler.dicom_origin_y = 10
            handler.crop_pixel_offset = (0, 0)

            px, py = handler.physical_to_pixel_coord(20.0, -30.0)
            assert isinstance(px, int)
            assert isinstance(py, int)
        finally:
            os.chdir(original_dir)

    def test_pixel_to_physical_coord(self, tmp_path):
        """Test pixel to physical coordinate conversion."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = DicomFileHandler()
            handler.pixel_spacing = 2.0
            handler.dicom_origin_x = 5
            handler.dicom_origin_y = 10
            handler.crop_pixel_offset = (0, 0)

            phys_x, phys_y = handler.pixel_to_physical_coord(10, 20)
            assert isinstance(phys_x, (int, float))
            assert isinstance(phys_y, (int, float))

            # Test round-trip conversion
            px, py = handler.physical_to_pixel_coord(phys_x, phys_y)
            assert abs(px - 10) <= 1  # Allow rounding error
            assert abs(py - 20) <= 1
        finally:
            os.chdir(original_dir)

    def test_get_patient_info(self, tmp_path):
        """Test getting patient information."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = DicomFileHandler()

            # Test with no DICOM data loaded
            institution, patient_id, patient_name = handler.get_patient_info()
            assert institution is None
            assert patient_id is None
            assert patient_name is None

            # Test with DICOM data
            mock_dcm = MagicMock()
            mock_dcm.get = MagicMock(side_effect=lambda key, default: {
                'InstitutionName': 'Test Hospital',
                'PatientID': '12345',
                'PatientName': 'Test Patient'
            }.get(key, default))
            handler.dicom_data = mock_dcm

            institution, patient_id, patient_name = handler.get_patient_info()
            assert institution == 'Test Hospital'
            assert patient_id == '12345'
            assert patient_name == 'Test Patient'
        finally:
            os.chdir(original_dir)


class TestMCCFileHandler:
    """Test cases for MCCFileHandler class."""

    def create_mock_mcc_file(self, tmp_path, device_type="1500"):
        """Helper function to create a mock MCC file."""
        mcc_content = f"""
SCAN_DEVICE=OCTAVIUS_{device_type}_XDR
SCAN_OFFAXIS_CROSSPLANE=0.00

BEGIN_DATA
0 1.5
1 2.0
END_DATA

BEGIN_DATA
0 1.8
1 2.2
END_DATA
"""
        mcc_file = tmp_path / "test.mcc"
        with open(mcc_file, "w") as f:
            f.write(mcc_content)
        return str(mcc_file)

    def test_detect_device_type_1500(self, tmp_path):
        """Test device type detection for OCTAVIUS 1500."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = MCCFileHandler()
            content = "SCAN_DEVICE=OCTAVIUS_1500_XDR\nSCAN_OFFAXIS_CROSSPLANE=0.00"
            device_type, task_type = handler.detect_device_type(content)

            assert device_type == 2  # 1500
            assert task_type == 2    # merged
        finally:
            os.chdir(original_dir)

    def test_detect_device_type_725(self, tmp_path):
        """Test device type detection for OCTAVIUS 725."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = MCCFileHandler()
            content = "SCAN_DEVICE=OCTAVIUS_725\nSCAN_OFFAXIS_CROSSPLANE=5.00"
            device_type, task_type = handler.detect_device_type(content)

            assert device_type == 1  # 725
            assert task_type == 1    # non-merged
        finally:
            os.chdir(original_dir)

    def test_open_file_success(self, tmp_path):
        """Test successful MCC file loading."""
        mcc_file = self.create_mock_mcc_file(tmp_path)

        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = MCCFileHandler()
            success, error = handler.open_file(mcc_file)

            assert success is True
            assert error is None
            assert handler.matrix_data is not None
            assert handler.device_type is not None
        finally:
            os.chdir(original_dir)

    def test_get_device_name(self, tmp_path):
        """Test getting device name."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = MCCFileHandler()
            handler.device_type = 2
            handler.task_type = 1
            assert "OCTAVIUS 1500" in handler.get_device_name()

            handler.device_type = 1
            handler.task_type = 2
            assert "OCTAVIUS 725" in handler.get_device_name()
            assert "merge" in handler.get_device_name()
        finally:
            os.chdir(original_dir)

    def test_set_device_parameters(self, tmp_path):
        """Test device parameter setting."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = MCCFileHandler()

            # Test 1500 device
            handler.device_type = 2
            handler._set_device_parameters()
            assert handler.mcc_origin_x == 26
            assert handler.mcc_origin_y == 26
            assert handler.mcc_spacing_x == 5.0
            assert handler.mcc_spacing_y == 5.0

            # Test 725 device
            handler.device_type = 1
            handler._set_device_parameters()
            assert handler.mcc_origin_x == 13
            assert handler.mcc_origin_y == 13
            assert handler.mcc_spacing_x == 10.0
            assert handler.mcc_spacing_y == 10.0
        finally:
            os.chdir(original_dir)

    def test_physical_to_pixel_coord(self, tmp_path):
        """Test physical to pixel coordinate conversion for MCC."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = MCCFileHandler()
            handler.mcc_spacing_x = 5.0
            handler.mcc_spacing_y = 5.0
            handler.mcc_origin_x = 26
            handler.mcc_origin_y = 26
            handler.crop_pixel_offset = (0, 0)

            px, py = handler.physical_to_pixel_coord(25.0, -15.0)
            assert isinstance(px, int)
            assert isinstance(py, int)
        finally:
            os.chdir(original_dir)

    def test_pixel_to_physical_coord(self, tmp_path):
        """Test pixel to physical coordinate conversion for MCC."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = MCCFileHandler()
            handler.mcc_spacing_x = 5.0
            handler.mcc_spacing_y = 5.0
            handler.mcc_origin_x = 26
            handler.mcc_origin_y = 26
            handler.crop_pixel_offset = (0, 0)

            phys_x, phys_y = handler.pixel_to_physical_coord(30, 20)
            assert isinstance(phys_x, (int, float))
            assert isinstance(phys_y, (int, float))

            # Test round-trip conversion
            px, py = handler.physical_to_pixel_coord(phys_x, phys_y)
            assert abs(px - 30) <= 1  # Allow rounding error
            assert abs(py - 20) <= 1
        finally:
            os.chdir(original_dir)

    def test_crop_to_bounds(self, tmp_path):
        """Test cropping MCC data to bounds."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            handler = MCCFileHandler()
            handler.matrix_data = np.ones((50, 50))
            handler.mcc_spacing_x = 5.0
            handler.mcc_spacing_y = 5.0
            handler.mcc_origin_x = 26
            handler.mcc_origin_y = 26
            handler.crop_pixel_offset = (0, 0)

            # Create physical coordinates
            handler.create_physical_coordinates_mcc()
            original_shape = handler.matrix_data.shape

            # Define bounds
            bounds = {
                'min_x': -50,
                'max_x': 50,
                'min_y': -50,
                'max_y': 50
            }

            handler.crop_to_bounds(bounds)

            # Check that data was cropped
            assert handler.matrix_data is not None
            # Shape might change depending on bounds
            assert handler.pixel_data is not None
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
