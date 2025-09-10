
import numpy as np
from scipy.interpolate import interpn
from file_handlers import DicomFileHandler, MCCFileHandler
from utils import logger

def calculate_gamma_manual(dose_ref, dose_eval, coords_y, coords_x, dta, dd, lower_dose_cutoff_percent):
    """
    수동으로 감마 인덱스를 계산합니다.
    
    Args:
        dose_ref: 참조 선량 배열
        dose_eval: 평가 선량 배열 (참조와 동일한 그리드)
        coords_y: y 좌표 벡터
        coords_x: x 좌표 벡터
        dta: DTA 기준 (mm)
        dd: DD 기준 (%)
        lower_dose_cutoff_percent: 선량 컷오프 (%)
        
    Returns:
        감마 맵 (numpy array)
    """
    gamma_map = np.full_like(dose_ref, np.nan)
    
    # DD 기준을 절대값으로 변환
    max_ref_dose = np.max(dose_ref)
    dd_threshold = (dd / 100.0) * max_ref_dose
    lower_dose_cutoff = (lower_dose_cutoff_percent / 100.0) * max_ref_dose

    # DTA에 해당하는 픽셀 수 계산
    pixel_spacing_y = np.abs(coords_y[1] - coords_y[0])
    pixel_spacing_x = np.abs(coords_x[1] - coords_x[0])
    dta_pixels_y = int(np.ceil(dta / pixel_spacing_y))
    dta_pixels_x = int(np.ceil(dta / pixel_spacing_x))

    # 참조 선량의 모든 픽셀에 대해 반복
    for y_ref, x_ref in np.ndindex(dose_ref.shape):
        
        # 선량 컷오프 확인
        if dose_ref[y_ref, x_ref] < lower_dose_cutoff:
            continue

        min_gamma = float('inf')

        # DTA 반경 내의 평가 픽셀 탐색
        y_min = max(0, y_ref - dta_pixels_y)
        y_max = min(dose_eval.shape[0], y_ref + dta_pixels_y + 1)
        x_min = max(0, x_ref - dta_pixels_x)
        x_max = min(dose_eval.shape[1], x_ref + dta_pixels_x + 1)

        found_pass = False
        for y_eval in range(y_min, y_max):
            for x_eval in range(x_min, x_max):
                
                # 거리 계산 (DTA)
                dist_sq = ((coords_y[y_ref] - coords_y[y_eval])**2 + 
                           (coords_x[x_ref] - coords_x[x_eval])**2)
                
                if dist_sq > dta**2:
                    continue

                # 선량 차이 계산 (DD)
                dose_diff = dose_eval[y_eval, x_eval] - dose_ref[y_ref, x_ref]
                
                # 감마 계산
                gamma_val_sq = (dist_sq / dta**2) + (dose_diff**2 / dd_threshold**2)
                
                if gamma_val_sq < min_gamma:
                    min_gamma = gamma_val_sq
                
                # 최적화: 감마값이 1보다 작은 지점을 찾으면 더 이상 탐색할 필요 없음
                if min_gamma <= 1:
                    found_pass = True
                    break
            if found_pass:
                break
        
        gamma_map[y_ref, x_ref] = np.sqrt(min_gamma)

    return gamma_map

def run_gamma_test_manual():
    """
    수동 감마 계산 로직을 테스트합니다.
    """
    # --- 1. 파일 경로 및 매개변수 설정 ---
    dicom_filepath = "1G240_2cm.dcm"
    mcc_filepath = "1G240_2cm.mcc"
    dose_percent_threshold = 3
    distance_mm_threshold = 3
    lower_percent_dose_cutoff = 10

    # --- 2. 파일 로드 ---
    logger.info("--- [수동 계산] 감마 분석 테스트 시작 ---")
    eval_handler = DicomFileHandler()
    eval_handler.open_file(dicom_filepath)
    ref_handler = MCCFileHandler()
    ref_handler.open_file(mcc_filepath)

    # --- 3. 데이터 추출 및 보간 ---
    dose_ref = ref_handler.get_matrix_data()
    ref_coords_y = ref_handler.phys_y_mesh[:, 0]
    ref_coords_x = ref_handler.phys_x_mesh[0, :]

    dose_eval_original = eval_handler.get_pixel_data()
    eval_coords_y_original = eval_handler.phys_y_mesh[:, 0]
    eval_coords_x_original = eval_handler.phys_x_mesh[0, :]

    points_to_interpolate = np.array(np.meshgrid(ref_coords_y, ref_coords_x, indexing='ij')).reshape(2, -1).T
    dose_eval_interp = interpn(
        (eval_coords_y_original, eval_coords_x_original),
        dose_eval_original, points_to_interpolate, method='linear',
        bounds_error=False, fill_value=0
    ).reshape(dose_ref.shape)

    logger.info("데이터 보간 완료.")

    # --- 4. 수동 감마 계산 ---
    logger.info("수동 감마 계산 시작...")
    gamma_map = calculate_gamma_manual(
        dose_ref, dose_eval_interp, 
        ref_coords_y, ref_coords_x, 
        distance_mm_threshold, dose_percent_threshold, lower_percent_dose_cutoff
    )
    logger.info("수동 감마 계산 완료.")

    # --- 5. 결과 분석 ---
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

    logger.info("--- [수동 계산] 감마 분석 테스트 종료 ---")

if __name__ == "__main__":
    run_gamma_test_manual()
