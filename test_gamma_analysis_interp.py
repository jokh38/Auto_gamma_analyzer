
import numpy as np
import pymedphys
from scipy.interpolate import interpn
from file_handlers import DicomFileHandler, MCCFileHandler
from utils import logger

def run_gamma_test_interpolated():
    """
    DICOM과 MCC 파일을 이용한 감마 분석 테스트 함수.
    Scipy로 DICOM 데이터를 MCC 좌표계에 맞춰 보간한 후 감마 분석을 수행합니다.
    """
    # --- 1. 파일 경로 설정 ---
    dicom_filepath = "1G240_2cm.dcm"
    mcc_filepath = "1G240_2cm.mcc"

    # --- 2. 핸들러 생성 및 파일 로드 ---
    logger.info("--- [보간 방식] 감마 분석 테스트 시작 ---")
    
    eval_handler = DicomFileHandler()
    eval_handler.open_file(dicom_filepath)
    logger.info(f"평가 파일 로드 완료: {dicom_filepath}")

    ref_handler = MCCFileHandler()
    ref_handler.open_file(mcc_filepath)
    logger.info(f"참조 파일 로드 완료: {mcc_filepath}")

    # --- 3. 데이터 추출 ---
    dose_reference = ref_handler.get_matrix_data()
    ref_coords_y = ref_handler.phys_y_mesh[:, 0]
    ref_coords_x = ref_handler.phys_x_mesh[0, :]
    axes_reference = (ref_coords_y, ref_coords_x)

    dose_evaluation_original = eval_handler.get_pixel_data()
    eval_coords_y_original = eval_handler.phys_y_mesh[:, 0]
    eval_coords_x_original = eval_handler.phys_x_mesh[0, :]

    # --- 4. DICOM 데이터를 MCC 좌표계로 보간 ---
    # MCC의 각 (y, x) 좌표 지점을 생성
    points_to_interpolate = np.array(np.meshgrid(ref_coords_y, ref_coords_x, indexing='ij'))
    # (2, N, M) -> (N*M, 2) 형태로 변환
    points_to_interpolate = points_to_interpolate.reshape(2, -1).T

    logger.info("Scipy.interpn을 사용하여 DICOM 데이터 보간 중...")
    
    # Scipy의 interpn을 사용하여 보간 수행
    dose_evaluation_interpolated = interpn(
        (eval_coords_y_original, eval_coords_x_original),
        dose_evaluation_original,
        points_to_interpolate,
        method='linear',
        bounds_error=False,
        fill_value=0
    ).reshape(dose_reference.shape)

    logger.info(f"보간 완료. 보간된 데이터 형태: {dose_evaluation_interpolated.shape}")

    # --- 5. 감마 분석 매개변수 설정 ---
    dose_percent_threshold = 3
    distance_mm_threshold = 3
    lower_percent_dose_cutoff = 10
    
    logger.info(f"감마 기준: DTA={distance_mm_threshold}mm, DD={dose_percent_threshold}%, Cutoff={lower_percent_dose_cutoff}%")

    # --- 6. 감마 분석 수행 (동일 좌표계 사용) ---
    try:
        # 이제 두 데이터셋이 동일한 좌표계를 가지므로, evaluation 축도 reference 축을 사용
        gamma_map = pymedphys.gamma(
            axes_reference,
            dose_reference,
            axes_reference,  # 동일한 좌표계 사용
            dose_evaluation_interpolated,
            dose_percent_threshold,
            distance_mm_threshold,
            lower_percent_dose_cutoff=lower_percent_dose_cutoff,
            local_gamma=False
        )

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

    logger.info("--- [보간 방식] 감마 분석 테스트 종료 ---")

if __name__ == "__main__":
    run_gamma_test_interpolated()
