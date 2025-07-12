import numpy as np
from file_handlers import DicomFileHandler, MCCFileHandler
from utils import logger

try:
    from gamma_index import gamma_matrix
    GAMMA_INDEX_AVAILABLE = True
except ImportError:
    GAMMA_INDEX_AVAILABLE = False

def run_gamma_test_with_library():
    """
    gamma_index 라이브러리를 이용한 감마 분석을 테스트합니다.
    """
    if not GAMMA_INDEX_AVAILABLE:
        logger.error("gamma_index 라이브러리를 찾을 수 없습니다.")
        logger.error("pip install gamma-index 를 실행하여 설치해주세요.")
        return

    # --- 1. 파일 경로 및 매개변수 설정 ---
    dicom_filepath = "1G240_2cm.dcm"
    mcc_filepath = "1G240_2cm.mcc"
    dose_percent_threshold = 3 # % 단위
    distance_mm_threshold = 3  # mm 단위
    lower_percent_dose_cutoff = 10 # % 단위

    # --- 2. 파일 로드 ---
    logger.info("--- [gamma_index 라이브러리] 테스트 시작 ---")
    eval_handler = DicomFileHandler()
    eval_handler.open_file(dicom_filepath)
    ref_handler = MCCFileHandler()
    ref_handler.open_file(mcc_filepath)

    # --- 3. 데이터 추출 및 보간 (MCC 그리드에 DICOM 맞추기) ---
    dose_ref = ref_handler.get_matrix_data()
    ref_coords_y = ref_handler.phys_y_mesh[:, 0]
    ref_coords_x = ref_handler.phys_x_mesh[0, :]
    
    dose_eval_original = eval_handler.get_pixel_data()
    eval_coords_y_original = eval_handler.phys_y_mesh[:, 0]
    eval_coords_x_original = eval_handler.phys_x_mesh[0, :]

    # Scipy를 사용하여 DICOM 데이터를 MCC 좌표계로 보간
    from scipy.interpolate import interpn
    points_to_interpolate = np.array(np.meshgrid(ref_coords_y, ref_coords_x, indexing='ij')).reshape(2, -1).T
    dose_eval_interp = interpn(
        (eval_coords_y_original, eval_coords_x_original),
        dose_eval_original, points_to_interpolate, method='linear',
        bounds_error=False, fill_value=0
    ).reshape(dose_ref.shape)

    logger.info("데이터 로드 및 보간 완료.")

    # --- 4. 선량 컷오프 및 유효하지 않은 MCC 값 적용 ---
    max_ref_dose = np.max(dose_ref[dose_ref != -1]) # -1이 아닌 유효한 값 중에서 최대값 찾기
    cutoff_value = (lower_percent_dose_cutoff / 100.0) * max_ref_dose
    
    dose_ref_masked = np.copy(dose_ref)
    dose_eval_masked = np.copy(dose_eval_interp)
    
    # MCC 데이터에서 -1인 지점을 NaN으로 마스킹
    invalid_mcc_mask = dose_ref_masked == -1
    dose_ref_masked[invalid_mcc_mask] = np.nan
    dose_eval_masked[invalid_mcc_mask] = np.nan # 해당 DICOM 보간 값도 함께 마스킹

    # 선량 컷오프 미만인 지점을 NaN으로 마스킹 (유효한 MCC 값 중에서)
    # 이미 invalid_mcc_mask로 NaN 처리된 부분은 이 마스크에 영향을 주지 않습니다.
    low_dose_mask = (dose_ref_masked < cutoff_value) & (~np.isnan(dose_ref_masked))
    dose_ref_masked[low_dose_mask] = np.nan
    dose_eval_masked[low_dose_mask] = np.nan

    # --- 5. DTA를 픽셀 단위로 변환 ---
    # MCC 핸들러에서 픽셀 간격 가져오기 (x, y 간격이 동일하다고 가정)
    pixel_spacing_x, pixel_spacing_y = ref_handler.get_spacing()
    # 여기서는 간단히 x, y 중 하나를 사용하거나 평균을 사용할 수 있습니다.
    # gamma_index는 픽셀 단위 DTA를 요구하므로, 실제 물리적 DTA를 픽셀 간격으로 나눕니다.
    dta_pixels = distance_mm_threshold / pixel_spacing_x # 또는 pixel_spacing_y

    # --- 6. 감마 계산 ---
    logger.info("gamma_index.gamma_matrix 함수 호출...")
    try:
        # gamma_matrix는 NaN 값을 자동으로 처리합니다.
        gamma_map = gamma_matrix(
            dose_ref_masked, # 마스킹된 참조 데이터
            dose_eval_masked, # 마스킹된 평가 데이터
            dta=dta_pixels, # 픽셀 단위 DTA
            dd=dose_percent_threshold / 100.0 # 상대값 DD
        )
        logger.info("감마 계산 성공.")

        # --- 7. 결과 분석 ---
        valid_gamma = gamma_map[~np.isnan(gamma_map)]
        if len(valid_gamma) == 0:
            logger.warning("감마 분석 결과, 유효한 지점이 없습니다.")
            pass_rate = 0
        else:
            pass_rate = np.sum(valid_gamma <= 1) / len(valid_gamma) * 100
        
        logger.info(f"총 {len(valid_gamma)}개 지점에서 감마 분석 수행")
        logger.info(f"감마 통과율 (<=1): {pass_rate:.2f}%")
        
        if pass_rate < 95:
            logger.warning(f"경고: 감마 통과율이 {pass_rate:.2f}%로 95% 미만입니다.")

    except Exception as e:
        logger.error(f"감마 분석 중 오류 발생: {e}", exc_info=True)

    logger.info("--- [gamma_index 라이브러리] 테스트 종료 ---")

if __name__ == "__main__":
    run_gamma_test_with_library()