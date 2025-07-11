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
        self.pixel_spacing = 1  # mm, 기본값
        self.physical_to_dicom_scale = 1.0
        self.dose_bounds = None
        
    def calculate_dose_bounds(self, dicom_image=None):
        """Calculate the bounds where dose is >= 10% of maximum"""
        if dicom_image is None:
            dicom_image = self.pixel_data
            
        if dicom_image is None:
            return None
            
        # Calculate 10% of maximum dose
        max_dose = np.max(dicom_image)
        threshold = 0.1 * max_dose
        
        # Create binary mask where dose >= threshold
        mask = dicom_image >= threshold
        
        # Find rows and columns with significant dose
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        # Find boundaries
        if not np.any(rows) or not np.any(cols):
            return None  # No significant dose found
            
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]
        
        # Add a small margin (5 pixels)
        margin = 5
        height, width = dicom_image.shape
        
        min_row = max(0, min_row - margin)
        max_row = min(height - 1, max_row + margin)
        min_col = max(0, min_col - margin)
        max_col = min(width - 1, max_col + margin)
        
        # Convert to physical coordinates
        min_x = self.dicom_origin_x * self.pixel_spacing + min_col * self.pixel_spacing
        max_x = self.dicom_origin_x * self.pixel_spacing + max_col * self.pixel_spacing
        min_y = self.dicom_origin_y * self.pixel_spacing + min_row * self.pixel_spacing
        max_y = self.dicom_origin_y * self.pixel_spacing + max_row * self.pixel_spacing
        
        # Return bounds in physical coordinates
        return {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'pixel_bounds': (min_row, max_row, min_col, max_col)
        }
        
    def update_origin(self, new_origin_x, new_origin_y):
        """원점 좌표 업데이트 및 물리적 좌표계 재계산"""
        self.dicom_origin_x = new_origin_x
        self.dicom_origin_y = new_origin_y
        self.create_physical_coordinates()
        self.dose_bounds = self.calculate_dose_bounds()
        logger.info(f"DICOM 원점 업데이트: ({self.dicom_origin_x}, {self.dicom_origin_y})")
        
    def open_file(self, filename):
        """DICOM RT dose 파일 로드"""
        try:
            self.filename = filename
            # pydicom으로 DICOM 파일 로드
            self.dicom_data = pydicom.dcmread(filename)
            
            # RT dose 파일인지 확인
            if self.dicom_data.Modality != 'RTDOSE':
                error_msg = "Selected file is not an RT Dose file"
                logger.error(error_msg)
                return False, error_msg
                
            # 픽셀 데이터 가져오기
            self.pixel_data = self.dicom_data.pixel_array * self.dicom_data.DoseGridScaling
            
            # 픽셀 간격 정보 가져오기 (가능한 경우)
            if hasattr(self.dicom_data, 'PixelSpacing'):
                self.pixel_spacing = float(self.dicom_data.PixelSpacing[0])  # mm
                self.physical_to_dicom_scale = 1.0 / self.pixel_spacing
                logger.info(f"DICOM 픽셀 간격: {self.pixel_spacing} mm")
            
            # 이미지 크기
            height, width = self.pixel_data.shape
            
            # ImagePositionPatient에서 원점 정보 설정
            if hasattr(self.dicom_data, 'ImagePositionPatient'):
                # DICOM 파일에서 실제 원점 정보 가져오기
                self.dicom_origin_x = int(self.dicom_data.ImagePositionPatient[0])
                self.dicom_origin_y = int(self.dicom_data.ImagePositionPatient[2])
                logger.info(f"DICOM 원점 설정: ({self.dicom_origin_x}, {self.dicom_origin_y}) 픽셀")
            else:
                # 기본값 설정
                self.dicom_origin_x = width // 2
                self.dicom_origin_y = height // 2
                logger.info(f"DICOM 원점 기본값 설정: ({self.dicom_origin_x}, {self.dicom_origin_y}) 픽셀")
                
            # 물리적 좌표계 생성
            self.create_physical_coordinates()
            
            # 도스 경계 계산
            self.dose_bounds = self.calculate_dose_bounds()
            
            return True
            
        except Exception as e:
            error_msg = f"DICOM 파일 로드 오류: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
            
    def create_physical_coordinates(self):
        """물리적 좌표계 생성"""
        if self.pixel_data is None:
            return
            
        height, width = self.pixel_data.shape
        
        # 물리적 좌표계 생성 (meshgrid 사용)
        phys_x = np.linspace(
            self.dicom_origin_x * self.pixel_spacing,
            (width + self.dicom_origin_x) * self.pixel_spacing,
            width
        )
        
        phys_y = np.linspace(
            self.dicom_origin_y * self.pixel_spacing,                 
            (height + self.dicom_origin_y) * self.pixel_spacing,
            height
        )
        
        # Meshgrid 생성
        self.phys_x_mesh, self.phys_y_mesh = np.meshgrid(phys_x, phys_y)
        
        # 물리적 범위 계산
        self.physical_extent = [phys_x.min(), phys_x.max(), phys_y.min(), phys_y.max()]
            
    def physical_to_pixel_coord(self, phys_x, phys_y):
        """물리적 좌표(mm)를 픽셀 좌표로 변환"""
        dicom_x = int(round(self.dicom_origin_x + phys_x / self.pixel_spacing))
        dicom_y = int(round(self.dicom_origin_y - phys_y / self.pixel_spacing))
        return dicom_x, dicom_y
    
    def pixel_to_physical_coord(self, dicom_x, dicom_y):
        """픽셀 좌표를 물리적 좌표(mm)로 변환"""
        phys_x = (dicom_x - self.dicom_origin_x) * self.pixel_spacing
        phys_y = (self.dicom_origin_y - dicom_y) * self.pixel_spacing
        return phys_x, phys_y

    def get_origin_coords(self):
        """원점 좌표 반환"""
        return self.dicom_origin_x, self.dicom_origin_y


class MCCFileHandler(BaseFileHandler):
    """MCC 파일을 처리하는 클래스"""
    def __init__(self):
        super().__init__()
        self.matrix_data = None
        self.device_type = None
        self.task_type = None
        self.n_rows = None
        self.mcc_origin_x = None
        self.mcc_origin_y = None
        self.mcc_spacing_x = None
        self.mcc_spacing_y = None
                
    def get_matrix_data(self):
        """매트릭스 데이터 반환"""
        return self.matrix_data

    def get_interpolated_matrix_data(self, method='cubic'):
        """2D 보간으로 매트릭스 데이터의 빈 영역 채우기"""
        if self.matrix_data is None:
            return None

        # -1 값을 NaN으로 변경하여 비어있는 데이터 표시
        data = self.matrix_data.copy()
        data[data < 0] = np.nan

        # 유효한 데이터 포인트의 인덱스 가져오기
        valid_points_indices = np.where(~np.isnan(data))
        valid_points_values = data[valid_points_indices]

        if len(valid_points_values) < 4:
            # 보간을 수행하기에 데이터 포인트가 충분하지 않음
            return self.matrix_data 

        # 전체 그리드 생성
        grid_y, grid_x = np.mgrid[0:data.shape[0], 0:data.shape[1]]

        # griddata를 사용하여 보간 수행
        interpolated_data = griddata(
            np.array(list(zip(valid_points_indices[0], valid_points_indices[1]))),
            valid_points_values,
            (grid_y, grid_x),
            method=method
        )
        
        return interpolated_data        
    
    def open_file(self, filename):
        """MCC 파일 로드 및 분석"""
        try:
            self.filename = filename
            with open(filename, "r") as file:
                lines = file.read().split()
            
            # 장비 및 작업 유형 감지
            self.device_type, self.task_type = self.detect_device_type(lines)
            
            # 데이터 추출
            N_begin = lines.count("BEGIN_DATA")
            self.n_rows = N_begin
            self.matrix_data = self.extract_data(lines, N_begin, self.device_type, self.task_type)
            
            # 원점 및 간격 정보 설정
            self._set_device_parameters()
            
            # 물리적 좌표계 생성
            self.create_physical_coordinates()
            
            logger.info(f"MCC 파일 로드 완료: {self.get_device_name()}")
            
            return True
                        
        except Exception as e:
            error_msg = f"File open error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _set_device_parameters(self):
        """장비 유형에 따른 파라미터 설정"""
        if self.device_type == 2:  # 1500
            self.mcc_origin_x = 27
            self.mcc_origin_y = 27
            self.mcc_spacing_x = 5
            self.mcc_spacing_y = 5
        else:  # 725
            self.mcc_origin_x = 13
            self.mcc_origin_y = 13
            self.mcc_spacing_x = 10
            self.mcc_spacing_y = 10
    
    def extract_data(self, lines, N_begin, device_type, task_type):
        """파일에서 측정 데이터 추출"""
        try:
            # 인터벌 설정
            if task_type == 1:
                tmp_intv = 1
                oct_read_intv = 3
            else:
                tmp_intv = 0.5
                oct_read_intv = 2
            
            # 데이터 구간 인덱스 찾기
            st_idx_j = []
            end_idx_j = []
            
            for j, line in enumerate(lines):
                if line == "BEGIN_DATA":
                    st_idx_j.append(j)
                elif line == "END_DATA":
                    end_idx_j.append(j)
            
            # 구간 정보 저장
            delimt_ind = np.zeros((len(st_idx_j), 3), dtype=int)
            delimt_ind[:, 0] = st_idx_j
            delimt_ind[:, 1] = end_idx_j
            delimt_ind[:, 2] = delimt_ind[:, 1] - delimt_ind[:, 0] - 1
            
            # 측정 데이터 저장
            matrix_octavius_mat_tmp = np.zeros((N_begin, N_begin)) - 1
            
            # 실제 x 방향 측정 지점 수
            x_lngt = int(delimt_ind[0, 2] / oct_read_intv)
            
            # 장비 유형과 작업 유형에 따른 데이터 추출
            if device_type == 2 and task_type == 1:
                # 1500 장비 타입 (F_OCTAV_read1500.m 로직)
                for j in range(0, N_begin, 2):  # 시작 -130
                    for k in range(x_lngt-1):
                        matrix_octavius_mat_tmp[j, 2*k+1] = float(lines[delimt_ind[j, 0] + 2 + oct_read_intv*(k)])
                
                for j in range(1, N_begin, 2):  # 시작 -125
                    for k in range(1, x_lngt):
                        matrix_octavius_mat_tmp[j, 2*k] = float(lines[delimt_ind[j, 0] + 2 + oct_read_intv*(k-1)])
            else:
                # 729 장비 타입 또는 기타 (F_OCTAV_read.m 로직)
                for j in range(N_begin):
                    for k in range(x_lngt):
                        matrix_octavius_mat_tmp[j, k] = float(lines[delimt_ind[j, 0] + 2 + oct_read_intv*k])
            
            # 상하 반전 적용
            matrix_octavius_mat = np.flipud(matrix_octavius_mat_tmp)
            
            return matrix_octavius_mat
        
        except Exception as e:
            logger.error(f"Data extraction error: {str(e)}")
            raise
        
    def detect_device_type(self, lines):
        """파일에서 장비 및 작업 유형 감지"""
        try:
            is_1500 = "SCAN_DEVICE=OCTAVIUS_1500_XDR" in lines
            is_merged = "SCAN_OFFAXIS_CROSSPLANE=0.00" in lines
            
            device_type = 2 if is_1500 else 1
            task_type = 2 if is_merged else 1
            
            return device_type, task_type
        except Exception as e:
            logger.error(f"Device type detection error: {str(e)}")
            raise

    def get_device_name(self):
        """장비 유형별 이름 반환"""
        if self.device_type == 2:
            if self.task_type == 1:
                return "OCTAVIUS 1500"
            else:
                return "OCTAVIUS 1500 with merge"
        else:
            if self.task_type == 1:
                return "OCTAVIUS 725"
            else:
                return "OCTAVIUS 725 with merge"
                
    def get_origin_coords(self):
        """장비 유형별 원점 좌표 반환"""
        return self.mcc_origin_x, self.mcc_origin_y    
            
    def get_spacing(self):
        """장비 유형별 간격 정보 반환"""
        return self.mcc_spacing_x, self.mcc_spacing_y

    def create_physical_coordinates(self):
        """물리적 좌표계 생성"""
        if self.matrix_data is None:
            return
            
        height, width = self.matrix_data.shape
        
        # 물리적 좌표계 생성 (meshgrid 사용)
        phys_x = np.linspace(
            -self.mcc_origin_x * self.mcc_spacing_x,
            (width - self.mcc_origin_x) * self.mcc_spacing_x,
            width
        )
        
        phys_y = np.linspace(
            (self.mcc_origin_y - height) * self.mcc_spacing_y,
            self.mcc_origin_y * self.mcc_spacing_y,
            height
        )
        
        # Meshgrid 생성
        self.phys_x_mesh, self.phys_y_mesh = np.meshgrid(phys_x, phys_y)
        
        # 물리적 범위 계산
        self.physical_extent = [phys_x.min(), phys_x.max(), phys_y.min(), phys_y.max()]
            
    def physical_to_pixel_coord(self, phys_x, phys_y):
        """물리적 좌표(mm)를 픽셀 좌표로 변환"""
        mcc_x = int(round(self.mcc_origin_x + phys_x / self.mcc_spacing_x))
        # For y-axis: top is positive, bottom is negative
        mcc_y = int(round(self.mcc_origin_y - phys_y / self.mcc_spacing_y))
        return mcc_x, mcc_y
    
    def pixel_to_physical_coord(self, mcc_x, mcc_y):
        """픽셀 좌표를 물리적 좌표(mm)로 변환"""
        phys_x = (mcc_x - self.mcc_origin_x) * self.mcc_spacing_x
        # For y-axis: top is positive, bottom is negative
        phys_y = (self.mcc_origin_y - mcc_y) * self.mcc_spacing_y
        return phys_x, phys_y
