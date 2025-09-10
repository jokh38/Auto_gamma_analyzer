
import numpy as np
import pymedphys
from scipy.interpolate import RegularGridInterpolator
from file_handlers import DicomFileHandler, MCCFileHandler
from utils import logger

def verify_gamma_at_mcc_points():
    """
    MCC 파일의 유효 측정점의 물리적 좌표를 기준으로 DICOM 선량을 비교하여 감마 분석을 수행합니다.
    이는 좌표계 일치 여부를 확인하기 위한 테스트입니다.
    """
    # --- 1. 파일 경로 설정 ---
    dicom_filepath = "1G240_2cm.dcm"
    mcc_filepath = "1G240_2cm.mcc"

    logger.info("--- MCC 측정점 기준 감마 분석 테스트 시작 ---")

    # --- 2. 핸들러 생성 및 파일 로드 ---
    # 평가(Evaluation) 데이터: DICOM
    eval_handler = DicomFileHandler()
    if not eval_handler.open_file(dicom_filepath):
        logger.error("DICOM 파일 로드 실패")
        return
    logger.info(f"평가 파일 로드 완료: {dicom_filepath}")

    # 참조(Reference) 데이터: MCC
    ref_handler = MCCFileHandler()
    if not ref_handler.open_file(mcc_filepath):
        logger.error("MCC 파일 로드 실패")
        return
    logger.info(f"참조 파일 로드 완료: {mcc_filepath}")

    # --- 3. 참조 데이터 준비 (MCC) ---
    mcc_dose_data = ref_handler.get_matrix_data()
    
    # 유효한 측정점의 인덱스 찾기 (값이 -1이 아닌 곳)
    valid_indices = np.where(mcc_dose_data >= 0)
    
    if valid_indices[0].size == 0:
        logger.error("MCC 파일에 유효한 측정 데이터가 없습니다.")
        return

    # 유효한 측정점의 픽셀 좌표 (y, x 순서)
    valid_pixel_coords_mcc = np.vstack(valid_indices).T

    # 물리적 좌표로 변환
    # pixel_to_physical_coord는 (x, y) 순서로 인자를 받으므로 순서 변경
    coords_ref = np.array([
        ref_handler.pixel_to_physical_coord(px, py) 
        for py, px in valid_pixel_coords_mcc
    ])
    
    # 유효한 측정점의 선량 값
    dose_ref = mcc_dose_data[valid_indices]
    
    logger.info(f"MCC에서 {len(dose_ref)}개의 유효 측정점을 찾았습니다.")

    # --- 4. 평가 데이터 준비 (DICOM) ---
    dicom_dose_data = eval_handler.get_pixel_data()
    
    # DICOM의 물리적 좌표 축 생성
    # DICOM 핸들러의 phys_x_mesh와 phys_y_mesh는 2D 그리드이므로 1D 축으로 변환
    dicom_phys_x_axis = eval_handler.phys_x_mesh[0, :]
    dicom_phys_y_axis = eval_handler.phys_y_mesh[:, 0]

    # Scipy를 이용한 2D 보간 함수 생성
    # RegularGridInterpolator는 좌표 축이 단조 증��해야 하므로, 필요한 경우 뒤집어 줍니다.
    if dicom_phys_y_axis[0] > dicom_phys_y_axis[-1]:
        dicom_phys_y_axis = np.flip(dicom_phys_y_axis)
        dicom_dose_data = np.flip(dicom_dose_data, axis=0)

    if dicom_phys_x_axis[0] > dicom_phys_x_axis[-1]:
        dicom_phys_x_axis = np.flip(dicom_phys_x_axis)
        dicom_dose_data = np.flip(dicom_dose_data, axis=1)

    interp_dicom = RegularGridInterpolator(
        (dicom_phys_y_axis, dicom_phys_x_axis), 
        dicom_dose_data,
        bounds_error=False, 
        fill_value=0
    )

    # MCC 측정점의 물리적 좌표(coords_ref)를 사용하여 DICOM 선량 보간
    # coords_ref는 (x, y) 순서이므로, 보간 함수에 맞게 (y, x) 순서로 전달
    dose_eval_interp = interp_dicom(coords_ref[:, ::-1])

    # --- 5. 감마 분석 매개변수 설정 ---
    dose_percent_threshold = 1  # 1%
    distance_mm_threshold = 1   # 1mm
    lower_percent_dose_cutoff = 10 # 최대 선량의 10% 이하 영역은 무시
    
    # 최대 선량은 참조(MCC) 선량을 기준으로 계산
    max_dose = np.max(dose_ref)
    
    logger.info(f"감마 기준: DTA={distance_mm_threshold}mm, DD={dose_percent_threshold}%, Cutoff={lower_percent_dose_cutoff}% of max_ref_dose({max_dose:.2f})")

    # --- 6. 감마 분석 수행 ---
    # pymedphys.gamma�� coords_eval 인자가 없으므로, 점별(point-wise) 감마를 직접 계산해야 함
    # gamma_index 라이브러리가 있다면 아래와 같이 사용 가능 (라이브러리 이름이 gamma_index라고 가정)
    # 여기서는 pymedphys의 내부 함수를 활용하여 점별 감마를 계산하는 방식을 모방합니다.
    
    try:
        # pymedphys.gamma는 전체 그리드를 비교하므로, 점별 비교를 위해 직접 계산
        # 여기서는 개념 증명을 위해 pymedphys.gamma를 호출하되,
        # coords_ref와 dose_ref, 그리고 보간된 dose_eval을 사용합니다.
        # pymedphys는 coords_eval을 직접 받지 않으므로, 동일한 좌표에 대한 점대점 비교를 수행합니다.
        
        gamma = pymedphys.gamma(
            coords_ref, dose_ref,
            coords_ref, dose_eval_interp, # 동일한 좌표, 다른 선량
            dose_percent_threshold,
            distance_mm_threshold,
            lower_percent_dose_cutoff=lower_percent_dose_cutoff,
            local_gamma=False, # Global gamma
            max_gamma=2, # 계산 속도를 위한 감마 상한 설정
            random_subset=None # 모든 점 계산
        )

        # --- 7. 결과 분석 ---
        valid_gamma = gamma[~np.isnan(gamma)]
        if len(valid_gamma) == 0:
            logger.warning("���마 분석 결과, 유효한 지점이 없습니다. (모든 지점이 Cutoff 이하일 수 있음)")
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

    logger.info("--- MCC 측정점 기준 감마 분석 테스트 종료 ---")


if __name__ == "__main__":
    verify_gamma_at_mcc_points()
