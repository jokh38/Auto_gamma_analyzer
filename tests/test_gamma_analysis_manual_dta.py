

import numpy as np
from file_handlers import DicomFileHandler, MCCFileHandler
from utils import logger

def calculate_gamma_manual_dta():
    """
    제안된 수동 DTA 검색 방식을 사용하여 감마 분석을 수행합니다.
    - 기준: MCC의 각 유효 측정점
    - 대상: DICOM의 조밀한 그리드
    - 방식: 각 MCC 점을 기준으로 DTA 반경 내 모든 DICOM 점을 검색하여
            선량 차이가 최소인 점을 찾아 감마를 계산합니다.
    """
    # --- 1. 파일 경로 및 매개변수 설정 ---
    dicom_filepath = "1G240_2cm.dcm"
    mcc_filepath = "1G240_2cm.mcc"
    dose_percent_threshold = 3  # %
    distance_mm_threshold = 3   # mm
    lower_percent_dose_cutoff = 10 # %

    logger.info("--- [수동 DTA 방식] 감마 분석 테스트 시작 ---")

    # --- 2. 파일 로드 ---
    eval_handler = DicomFileHandler()
    eval_handler.open_file(dicom_filepath)
    ref_handler = MCCFileHandler()
    ref_handler.open_file(mcc_filepath)

    # --- 3. 데이터 준비 및 좌표 정렬 ---
    # 3-1. MCC 유효 측정점 추출
    mcc_dose_data = ref_handler.get_matrix_data()
    valid_indices = np.where(mcc_dose_data >= 0)
    coords_ref_phys = np.array([ref_handler.pixel_to_physical_coord(px, py) for py, px in np.vstack(valid_indices).T])
    dose_ref_points = mcc_dose_data[valid_indices]
    
    if dose_ref_points.size == 0:
        logger.error("MCC 파일에 유효한 측정 데이터가 없습니다.")
        return

    # 3-2. DICOM 데이터 준비
    dicom_dose_data = eval_handler.get_pixel_data()

    # 3-3. 최대 선량점 기준 좌표 정렬
    max_dose_idx_mcc = np.argmax(dose_ref_points)
    max_dose_phys_coord_mcc = coords_ref_phys[max_dose_idx_mcc]
    
    max_dose_pixel_idx_dicom = np.unravel_index(np.argmax(dicom_dose_data), dicom_dose_data.shape)
    max_dose_phys_coord_dicom = eval_handler.pixel_to_physical_coord(max_dose_pixel_idx_dicom[1], max_dose_pixel_idx_dicom[0])

    shift = np.array(max_dose_phys_coord_dicom) - np.array(max_dose_phys_coord_mcc)
    coords_ref_phys_shifted = coords_ref_phys + shift
    logger.info(f"좌표 정렬 완료 (이동량: dx={shift[0]:.2f}, dy={shift[1]:.2f} mm)")

    # --- 4. 감마 계산 ---
    gamma_values = []
    max_ref_dose = np.max(dose_ref_points)
    dd_abs_threshold = (dose_percent_threshold / 100.0) * max_ref_dose
    cutoff_value = (lower_percent_dose_cutoff / 100.0) * max_ref_dose

    points_analyzed = 0
    for i in range(len(coords_ref_phys_shifted)):
        coord_ref = coords_ref_phys_shifted[i]
        dose_ref = dose_ref_points[i]

        # 4-1. 선량 컷오프 적용
        if dose_ref < cutoff_value:
            continue
        points_analyzed += 1

        # 4-2. DTA 반경 내 DICOM 후보 픽셀 검색
        # 검색할 사각 영역의 픽셀 범위 계산
        min_x_search, min_y_search = eval_handler.physical_to_pixel_coord(coord_ref[0] - distance_mm_threshold, coord_ref[1] - distance_mm_threshold)
        max_x_search, max_y_search = eval_handler.physical_to_pixel_coord(coord_ref[0] + distance_mm_threshold, coord_ref[1] + distance_mm_threshold)
        
        candidate_pixels = []
        for r in range(min_y_search, max_y_search + 1):
            for c in range(min_x_search, max_x_search + 1):
                try:
                    phys_coord_cand = eval_handler.pixel_to_physical_coord(c, r)
                    dist_sq = (phys_coord_cand[0] - coord_ref[0])**2 + (phys_coord_cand[1] - coord_ref[1])**2
                    
                    if dist_sq <= distance_mm_threshold**2:
                        dose_cand = dicom_dose_data[r, c]
                        candidate_pixels.append({'dist_sq': dist_sq, 'dose_diff': abs(dose_cand - dose_ref), 'dist': np.sqrt(dist_sq), 'dose': dose_cand})
                except IndexError:
                    continue # DICOM 그리드 범위를 벗어나는 경우 무시

        # 4-3. 최적의 후보 탐색 및 감마 계산
        if not candidate_pixels:
            gamma_val = np.inf # 후보가 없으면 감마는 무한대
        else:
            # 선량 차이가 가장 작은 후보를 찾음
            best_candidate = min(candidate_pixels, key=lambda x: x['dose_diff'])
            
            # 감마 계산
            gamma_sq = (best_candidate['dist'] / distance_mm_threshold)**2 + \
                       (best_candidate['dose_diff'] / dd_abs_threshold)**2
            gamma_val = np.sqrt(gamma_sq)
        
        gamma_values.append(gamma_val)

    # --- 5. 결과 분석 ---
    gamma_values = np.array(gamma_values)
    valid_gamma = gamma_values[~np.isinf(gamma_values)]

    if len(valid_gamma) == 0:
        logger.warning("감마 분석 결과, 유효한 지점이 없습니다.")
        pass_rate = 0
    else:
        pass_rate = np.sum(valid_gamma <= 1) / len(valid_gamma) * 100
    
    logger.info(f"총 {points_analyzed}개 지점에서 감마 분석 수행 (컷오프 적용 후)")
    logger.info(f"감마 통과율 (<=1): {pass_rate:.2f}%")
    
    if pass_rate < 95:
        logger.warning(f"경고: 감마 통과율이 95% 미만입니다. (통과율: {pass_rate:.2f}%)")
    else:
        logger.info(f"테스트 통과: 감마 통과율이 95% 이상입니다. (통과율: {pass_rate:.2f}%)")

    logger.info("--- [수동 DTA 방식] 감마 분석 테스트 종료 ---")


if __name__ == "__main__":
    calculate_gamma_manual_dta()

