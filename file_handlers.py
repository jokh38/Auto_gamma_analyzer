import os
import numpy as np
import pydicom
from scipy.interpolate import griddata
from utils import logger

class BaseFileHandler:
    """다양한 파일 핸들러를 위한 기본 클래스"""
    def __init__(self):
        self.filename = None
        self.pixel_data = None
        self.phys_x_mesh = None
        self.phys_y_mesh = None
        self.physical_extent = None
        self.origin_x = 0
        self.origin_y = 0
        self.pixel_spacing = 1.0
        
    def get_filename(self):
        """파일명 반환"""
        if self.filename:
            return os.path.basename(self.filename)
        return None
    
    def get_physical_extent(self):
        """물리적 좌표 범위 반환"""
        return self.physical_extent
    
    def get_origin_coords(self):
        """원점 좌표 반환"""
        return self.origin_x, self.origin_y
        
    def get_spacing(self):
        """픽셀 간격 반환"""
        return self.pixel_spacing, self.pixel_spacing
    
    def get_pixel_data(self):
        """픽셀 데이터 반환"""
        return self.pixel_data
    
    def create_physical_coordinates(self):
        """물리적 좌표계 생성 (추상 메서드)"""
        raise NotImplementedError("서브클래스에서 구현해야 함")
    
    def physical_to_pixel_coord(self, phys_x, phys_y):
        """물리적 좌표(mm)를 픽셀 좌표로 변환 (추상 메서드)"""
        raise NotImplementedError("서브클래스에서 구현해야 함")
    
    def pixel_to_physical_coord(self, pixel_x, pixel_y):
        """픽셀 좌표를 물리적 좌표(mm)로 변환 (추상 메서드)"""
        raise NotImplementedError("서브클래스에서 구현해야 함")
        
    def open_file(self, filename):
        """파일 로드 (추상 메서드)"""
        try:
            self.filename = filename
            # 파일 로드 로직 (서브클래스에서 구현)
            return True
        except Exception as e:
            error_msg = f"파일 로드 오류: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


class DicomFileHandler(BaseFileHandler):
    """DICOM RT dose 파일을 처리하는 클래스"""
    def __init__(self):
        super().__init__()
        self.dicom_data = None
        self.dicom_origin_x = 0
        self.dicom_origin_y = 0
        self.pixel_spacing = 1.0
        self.dose_bounds = None
        
    def open_file(self, filename):
        """DICOM RT dose 파일 로드"""
        try:
            self.filename = filename
            self.dicom_data = pydicom.dcmread(filename)
            
            if self.dicom_data.Modality != 'RTDOSE':
                raise ValueError("Selected file is not an RT Dose file")
                
            self.pixel_data = self.dicom_data.pixel_array * self.dicom_data.DoseGridScaling
            
            if hasattr(self.dicom_data, 'PixelSpacing'):
                self.pixel_spacing = float(self.dicom_data.PixelSpacing[0])
            
            height, width = self.pixel_data.shape
            
            if hasattr(self.dicom_data, 'ImagePositionPatient'):
                # ImagePositionPatient: [x, z, y]
                # 물리적 위치를 픽셀 간격으로 나누어 픽셀 원점 계산
                self.dicom_origin_x = int(round(self.dicom_data.ImagePositionPatient[0] / self.pixel_spacing))
                self.dicom_origin_y = int(round(self.dicom_data.ImagePositionPatient[2] / self.pixel_spacing)) # y좌표는 index 2
                
                logger.info(f"DICOM 픽셀 원점 설정 (from ImagePositionPatient): x={self.dicom_origin_x}, y={self.dicom_origin_y}")
                logger.info(f"사용된 ImagePositionPatient 값: x={self.dicom_data.ImagePositionPatient[0]}, y={self.dicom_data.ImagePositionPatient[2]}")
            else:
                self.dicom_origin_x = -width // 2
                self.dicom_origin_y = -height // 2
                logger.warning(f"DICOM 원점 정보 없음. 이미지 중심으로 기본값 설정: ({self.dicom_origin_x}, {self.dicom_origin_y})")
                
            self.create_physical_coordinates()
            self.dose_bounds = self.calculate_dose_bounds() # 자동 크롭 기능 활성화
            
            return True
            
        except Exception as e:
            error_msg = f"DICOM 파일 로드 오류: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def calculate_dose_bounds(self, dicom_image=None, threshold_percent=10, margin_mm=20):
        """최대 선량의 특정 비율 이상인 영역의 경계를 계산합니다."""
        if dicom_image is None:
            dicom_image = self.get_pixel_data()
        if dicom_image is None:
            return None

        max_dose = np.max(dicom_image)
        threshold_val = (threshold_percent / 100.0) * max_dose
        
        mask = dicom_image >= threshold_val
        if not np.any(mask):
            return None

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]
        
        # 픽셀 좌표를 물리적 좌표로 변환
        min_phys_x, min_phys_y = self.pixel_to_physical_coord(min_col, min_row)
        max_phys_x, max_phys_y = self.pixel_to_physical_coord(max_col, max_row)
        
        # 물리적 좌표에 여백(margin) 추가
        min_phys_x -= margin_mm
        max_phys_x += margin_mm
        min_phys_y -= margin_mm
        max_phys_y += margin_mm
        
        bounds = {
            'min_x': min_phys_x, 'max_x': max_phys_x,
            'min_y': min_phys_y, 'max_y': max_phys_y
        }
        logger.info(f"계산된 선량 경계 (물리적 좌표, {margin_mm}mm 여백 포함): {bounds}")
        return bounds

    def create_physical_coordinates(self):
        if self.pixel_data is None: return
        height, width = self.pixel_data.shape
        phys_x = (np.arange(width) + self.dicom_origin_x) * self.pixel_spacing
        phys_y = (np.arange(height) + self.dicom_origin_y) * self.pixel_spacing
        self.phys_x_mesh, self.phys_y_mesh = np.meshgrid(phys_x, phys_y)
        self.physical_extent = [phys_x.min(), phys_x.max(), phys_y.min(), phys_y.max()]

    def physical_to_pixel_coord(self, phys_x, phys_y):
        pixel_x = int(round(phys_x / self.pixel_spacing - self.dicom_origin_x))
        pixel_y = int(round(phys_y / self.pixel_spacing - self.dicom_origin_y))
        return pixel_x, pixel_y
    
    def pixel_to_physical_coord(self, pixel_x, pixel_y):
        phys_x = (pixel_x + self.dicom_origin_x) * self.pixel_spacing
        phys_y = (pixel_y + self.dicom_origin_y) * self.pixel_spacing
        return phys_x, phys_y

    def get_origin_coords(self):
        return self.dicom_origin_x, self.dicom_origin_y

    def get_spacing(self):
        return self.pixel_spacing, self.pixel_spacing

    def get_patient_info(self):
        """환자 정보(기관명, ID, 이름) 반환"""
        if self.dicom_data is None:
            return None, None, None

        institution = self.dicom_data.get("InstitutionName", "N/A")
        patient_id = self.dicom_data.get("PatientID", "N/A")
        patient_name = self.dicom_data.get("PatientName", "N/A")

        return institution, patient_id, patient_name

class MCCFileHandler(BaseFileHandler):
    """MCC 파일을 처리하는 클래스"""
    def __init__(self):
        super().__init__()
        self.matrix_data = None
        self.device_type = None
        self.task_type = None
        self.n_rows = None
        self.mcc_origin_x = 0
        self.mcc_origin_y = 0
        self.mcc_spacing_x = 1.0
        self.mcc_spacing_y = 1.0
                
    def get_matrix_data(self):
        return self.matrix_data

    def get_interpolated_matrix_data(self, method='cubic'):
        if self.matrix_data is None: return None
        data = self.matrix_data.copy()
        data[data < 0] = np.nan
        valid_points_indices = np.where(~np.isnan(data))
        if len(valid_points_indices[0]) < 4: return self.matrix_data 
        grid_y, grid_x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        interpolated_data = griddata(
            np.array(list(zip(valid_points_indices[0], valid_points_indices[1]))),
            data[valid_points_indices],
            (grid_y, grid_x),
            method=method
        )
        return interpolated_data        
    
    def open_file(self, filename):
        try:
            self.filename = filename
            with open(filename, "r") as file:
                lines = file.read().split()
            
            self.device_type, self.task_type = self.detect_device_type(lines)
            
            N_begin = lines.count("BEGIN_DATA")
            self.n_rows = N_begin
            self.matrix_data = self.extract_data(lines, N_begin, self.device_type, self.task_type)
            self.pixel_data = self.matrix_data
            
            self._set_device_parameters()
            self.create_physical_coordinates()
            
            logger.info(f"MCC 파일 로드 완료: {self.get_device_name()}")
            return True
                        
        except Exception as e:
            error_msg = f"File open error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _set_device_parameters(self):
        if self.device_type == 2:  # 1500
            self.mcc_origin_x = 27
            self.mcc_origin_y = 27
            self.mcc_spacing_x = 5.0
            self.mcc_spacing_y = 5.0
        else:  # 725
            self.mcc_origin_x = 13
            self.mcc_origin_y = 13
            self.mcc_spacing_x = 10.0
            self.mcc_spacing_y = 10.0
    
    def extract_data(self, lines, N_begin, device_type, task_type):
        # ... (이전과 동일한 데이터 추출 로직)
        try:
            if task_type == 1: oct_read_intv = 3
            else: oct_read_intv = 2
            st_idx_j, end_idx_j = [], []
            for j, line in enumerate(lines):
                if line == "BEGIN_DATA": st_idx_j.append(j)
                elif line == "END_DATA": end_idx_j.append(j)
            delimt_ind = np.zeros((len(st_idx_j), 3), dtype=int)
            delimt_ind[:, 0], delimt_ind[:, 1] = st_idx_j, end_idx_j
            delimt_ind[:, 2] = delimt_ind[:, 1] - delimt_ind[:, 0] - 1
            matrix_octavius_mat_tmp = np.zeros((N_begin, N_begin)) - 1
            x_lngt = int(delimt_ind[0, 2] / oct_read_intv)
            if device_type == 2 and task_type == 1:
                for j in range(0, N_begin, 2):
                    for k in range(x_lngt-1):
                        matrix_octavius_mat_tmp[j, 2*k+1] = float(lines[delimt_ind[j, 0] + 2 + oct_read_intv*(k)])
                for j in range(1, N_begin, 2):
                    for k in range(1, x_lngt):
                        matrix_octavius_mat_tmp[j, 2*k] = float(lines[delimt_ind[j, 0] + 2 + oct_read_intv*(k-1)])
            else:
                for j in range(N_begin):
                    for k in range(x_lngt):
                        matrix_octavius_mat_tmp[j, k] = float(lines[delimt_ind[j, 0] + 2 + oct_read_intv*k])
            return np.flipud(matrix_octavius_mat_tmp)
        except Exception as e:
            logger.error(f"Data extraction error: {str(e)}")
            raise
        
    def detect_device_type(self, lines):
        try:
            is_1500 = "SCAN_DEVICE=OCTAVIUS_1500_XDR" in lines
            is_merged = "SCAN_OFFAXIS_CROSSPLANE=0.00" in lines
            return (2 if is_1500 else 1), (2 if is_merged else 1)
        except Exception as e:
            logger.error(f"Device type detection error: {str(e)}")
            raise

    def get_device_name(self):
        if self.device_type == 2: return "OCTAVIUS 1500" + (" with merge" if self.task_type == 2 else "")
        else: return "OCTAVIUS 725" + (" with merge" if self.task_type == 2 else "")
                
    def get_origin_coords(self):
        return self.mcc_origin_x, self.mcc_origin_y
            
    def get_spacing(self):
        return self.mcc_spacing_x, self.mcc_spacing_y

    def create_physical_coordinates(self):
        if self.matrix_data is None: return
        height, width = self.matrix_data.shape
        phys_x = (np.arange(width) - self.mcc_origin_x) * self.mcc_spacing_x
        phys_y = (np.arange(height) - self.mcc_origin_y) * self.mcc_spacing_y
        self.phys_x_mesh, self.phys_y_mesh = np.meshgrid(phys_x, phys_y)
        self.physical_extent = [phys_x.min(), phys_x.max(), phys_y.min(), phys_y.max()]
            
    def physical_to_pixel_coord(self, phys_x, phys_y):
        pixel_x = int(round(phys_x / self.mcc_spacing_x + self.mcc_origin_x))
        pixel_y = int(round(phys_y / self.mcc_spacing_y + self.mcc_origin_y))
        return pixel_x, pixel_y
    
    def pixel_to_physical_coord(self, pixel_x, pixel_y):
        phys_x = (pixel_x - self.mcc_origin_x) * self.mcc_spacing_x
        phys_y = (pixel_y - self.mcc_origin_y) * self.mcc_spacing_y
        return phys_x, phys_y
