
import numpy as np
import pymedphys
from file_handlers import DicomFileHandler, MCCFileHandler
from utils import logger

def run_gamma_test():
    """
    DICOM과 MCC 파일을 이용한 감마 분석 테스트 함수.
    UI 없이 순수하게 데이터 처리 및 감마 분석 기능만 테스트합니다.
    """
    # --- 1. 파일 경로 설정 ---
    dicom_filepath = "1G240_2cm.dcm"
    mcc_filepath = "1G240_2cm.mcc"

    # --- 2. 핸들러 생성 및 파일 로드 ---
    logger.info("--- 감마 분석 테스트 시작 ---")
    
    # 평가(Evaluation) 데이터: DICOM
    eval_handler = DicomFileHandler()
    eval_handler.open_file(dicom_filepath)
    logger.info(f"평가 파일 로드 완료: {dicom_filepath}")

    # 참조(Reference) 데이터: MCC
    ref_handler = MCCFileHandler()
    ref_handler.open_file(mcc_filepath)
    logger.info(f"참조 파일 로드 완료: {mcc_filepath}")

    # --- 3. 데이터 추출 및 형식 맞추기 ---
    
    # 참조 데이터 (MCC)
    dose_reference = ref_handler.get_matrix_data()
    ref_phys_y = ref_handler.phys_y_mesh[:, 0]
    ref_phys_x = ref_handler.phys_x_mesh[0, :]
    axes_reference = (ref_phys_y, ref_phys_x)
    
    # 평가 데이터 (DICOM)
    dose_evaluation = eval_handler.get_pixel_data()
    eval_phys_y = eval_handler.phys_y_mesh[:, 0]
    eval_phys_x = eval_handler.phys_x_mesh[0, :]
    axes_evaluation = (eval_phys_y, eval_phys_x)

    logger.info(f"참조 데이터 형태: {dose_reference.shape}, 좌표 y: {len(ref_phys_y)}, x: {len(ref_phys_x)}")
    logger.info(f"평가 데이터 형태: {dose_evaluation.shape}, 좌표 y: {len(eval_phys_y)}, x: {len(eval_phys_x)}")

    # --- 4. 감마 분석 매개변수 설정 ---
    dose_percent_threshold = 3  # 3%
    distance_mm_threshold = 3   # 3mm
    lower_percent_dose_cutoff = 10 # 최대 선량의 10% 이하 영역은 무시
    
    logger.info(f"감마 기준: DTA={distance_mm_threshold}mm, DD={dose_percent_threshold}%, Cutoff={lower_percent_dose_cutoff}%")

    # --- 5. 감마 분석 수행 ---
    try:
        gamma_map = pymedphys.gamma(
            axes_reference,
            dose_reference,
            axes_evaluation,
            dose_evaluation,
            dose_percent_threshold,
            distance_mm_threshold,
            lower_percent_dose_cutoff=lower_percent_dose_cutoff,
            local_gamma=False # Global gamma
        )

        # --- 6. 결과 분석 ---
        valid_gamma = gamma_map[~np.isnan(gamma_map)]
        if len(valid_gamma) == 0:
            logger.warning("감마 분석 결과, 유효한 지점이 없습니다.")
            pass_rate = 0
        else:
            pass_rate = np.sum(valid_gamma <= 1) / len(valid_gamma) * 100
        
        logger.info(f"총 {len(valid_gamma)}개 지점에서 감마 분석 수행")
        logger.info(f"감마 통과율 (<=1): {pass_rate:.2f}%")
        
        if pass_rate < 95:
            logger.warning("경고: 감마 통과율이 95% 미만입니다.")

    except Exception as e:
        logger.error(f"감마 분석 중 오류 발생: {e}", exc_info=True)

    logger.info("--- 감마 분석 테스트 종료 ---")


if __name__ == "__main__":
    run_gamma_test()
