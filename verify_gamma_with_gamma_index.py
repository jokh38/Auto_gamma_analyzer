
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from file_handlers import DicomFileHandler, MCCFileHandler
from utils import logger

try:
    from gamma_index import gamma_matrix
    GAMMA_INDEX_AVAILABLE = True
except ImportError:
    GAMMA_INDEX_AVAILABLE = False

def verify_gamma_with_gamma_index():
    """
    gamma_index 라이브러리를 사용하여 MCC의 유효 측정점 기준 감마 분석을 수행합니다.
    MCC의 물리적 좌표에 해당하는 DICOM 선량을 보간하여 비교합니다.
    """
    if not GAMMA_INDEX_AVAILABLE:
        logger.error("gamma_index 라이브러리를 찾을 수 없습니다.")
        logger.error("터미널에서 'pip install gamma-index'를 실행하여 설치해주세요.")
        return

    # --- 1. 파일 경로 및 매개변수 설정 ---
    dicom_filepath = "1G240_2cm.dcm"
    mcc_filepath = "1G240_2cm.mcc"
    dose_percent_threshold = 3  # %
    distance_mm_threshold = 3   # mm
    lower_percent_dose_cutoff = 10 # %

    logger.info("--- [gamma_index] MCC 측정점 기준 감마 분석 테스트 시작 ---")

    # --- 2. 파일 로드 ---
    eval_handler = DicomFileHandler()
    eval_handler.open_file(dicom_filepath)
    ref_handler = MCCFileHandler()
    ref_handler.open_file(mcc_filepath)

    # --- 3. MCC 유효 측정점의 물리적 좌표 및 선량 추출 ---
    mcc_dose_data = ref_handler.get_matrix_data()
    valid_indices = np.where(mcc_dose_data >= 0)
    
    if valid_indices[0].size == 0:
        logger.error("MCC 파일에 유효한 측정 데이터가 없습니다.")
        return

    valid_pixel_coords_mcc = np.vstack(valid_indices).T
    coords_ref_phys = np.array([
        ref_handler.pixel_to_physical_coord(px, py) 
        for py, px in valid_pixel_coords_mcc
    ])
    dose_ref_points = mcc_dose_data[valid_indices]
    logger.info(f"MCC에서 {len(dose_ref_points)}개의 유효 측정점을 찾았습니다.")

    # --- 4. 좌표계 정렬 및 DICOM 데이터 보간 ---
    dicom_dose_data = eval_handler.get_pixel_data()

    # 4-1. 각 데이터에서 최대 선량점의 물리적 좌표 찾기
    # MCC
    max_dose_idx_mcc = np.argmax(dose_ref_points)
    max_dose_phys_coord_mcc = coords_ref_phys[max_dose_idx_mcc]
    
    # DICOM
    max_dose_pixel_idx_dicom = np.unravel_index(np.argmax(dicom_dose_data), dicom_dose_data.shape)
    max_dose_phys_coord_dicom = eval_handler.pixel_to_physical_coord(max_dose_pixel_idx_dicom[1], max_dose_pixel_idx_dicom[0])

    # 4-2. 좌표 이동(shift)량 계산
    shift_x = max_dose_phys_coord_dicom[0] - max_dose_phys_coord_mcc[0]
    shift_y = max_dose_phys_coord_dicom[1] - max_dose_phys_coord_mcc[1]
    logger.info(f"좌표 정렬을 위한 이동량 계산: dx={shift_x:.2f}mm, dy={shift_y:.2f}mm")

    # 4-3. MCC 좌표계 이동
    coords_ref_phys_shifted = coords_ref_phys + np.array([shift_x, shift_y])
    logger.info("MCC 물리적 좌표계를 이동하여 DICOM 최대 선량점에 정렬했습니다.")

    # 4-4. 이동된 좌표를 사용하여 DICOM 데이터 보간
    dicom_phys_x_axis = eval_handler.phys_x_mesh[0, :]
    dicom_phys_y_axis = eval_handler.phys_y_mesh[:, 0]

    if dicom_phys_y_axis[0] > dicom_phys_y_axis[-1]:
        dicom_phys_y_axis = np.flip(dicom_phys_y_axis)
        dicom_dose_data_flipped = np.flip(dicom_dose_data, axis=0)
    else:
        dicom_dose_data_flipped = dicom_dose_data

    if dicom_phys_x_axis[0] > dicom_phys_x_axis[-1]:
        dicom_phys_x_axis = np.flip(dicom_phys_x_axis)
        dicom_dose_data_flipped = np.flip(dicom_dose_data_flipped, axis=1)

    interp_dicom = RegularGridInterpolator(
        (dicom_phys_y_axis, dicom_phys_x_axis), 
        dicom_dose_data_flipped, bounds_error=False, fill_value=0
    )
    dose_eval_points_interp = interp_dicom(coords_ref_phys_shifted[:, ::-1])

    # --- 5. gamma_matrix를 위한 행렬 데이터 재구성 ---
    # 빈 행렬을 NaN으로 초기화
    dose_ref_matrix = np.full_like(mcc_dose_data, np.nan, dtype=np.float64)
    dose_eval_matrix = np.full_like(mcc_dose_data, np.nan, dtype=np.float64)

    # 유효한 지점에만 선량 값 채우기
    dose_ref_matrix[valid_indices] = dose_ref_points
    dose_eval_matrix[valid_indices] = dose_eval_points_interp

    # --- 6. 선량 컷오프 적용 ---
    max_ref_dose = np.nanmax(dose_ref_matrix)
    cutoff_value = (lower_percent_dose_cutoff / 100.0) * max_ref_dose
    
    low_dose_mask = dose_ref_matrix < cutoff_value
    dose_ref_matrix[low_dose_mask] = np.nan
    dose_eval_matrix[low_dose_mask] = np.nan
    
    logger.info(f"감마 기준: DTA={distance_mm_threshold}mm, DD={dose_percent_threshold}%, Cutoff={lower_percent_dose_cutoff}% ({cutoff_value:.2f})")

    # --- 7. DTA를 픽셀 단위로 변환 ---
    pixel_spacing_x, _ = ref_handler.get_spacing()
    dta_pixels = distance_mm_threshold / pixel_spacing_x
    logger.info(f"물리적 DTA {distance_mm_threshold}mm -> 픽셀 DTA {dta_pixels:.2f} pixels")

    # --- 8. 감마 계산 ---
    try:
        gamma_map = gamma_matrix(
            dose_ref_matrix,
            dose_eval_matrix,
            dta=dta_pixels,
            dd=dose_percent_threshold / 100.0
        )
        
        # --- 9. 결과 분석 ---
        valid_gamma = gamma_map[~np.isnan(gamma_map)]
        if len(valid_gamma) == 0:
            logger.warning("감마 분석 결과, 유효한 지점이 없습니다 (모든 지점이 Cutoff 이하일 수 있음).")
            pass_rate = 0
        else:
            pass_rate = np.sum(valid_gamma <= 1) / len(valid_gamma) * 100
        
        logger.info(f"총 {len(valid_gamma)}개 지점에서 감마 분석 수행")
        logger.info(f"감마 통과율 (<=1): {pass_rate:.2f}%")
        
        if pass_rate < 95:
            logger.warning(f"경고: 감마 통과율이 95% 미만입니다. (통과율: {pass_rate:.2f}%)")
        else:
            logger.info(f"테스트 통과: 감마 통과율이 95% 이상입니다. (통과율: {pass_rate:.2f}%)")

    except Exception as e:
        logger.error(f"감마 분석 중 오류 발생: {e}", exc_info=True)

    logger.info("--- [gamma_index] MCC 측정점 기준 감마 분석 테스트 종료 ---")

if __name__ == "__main__":
    verify_gamma_with_gamma_index()
