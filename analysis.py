import numpy as np
import pymedphys
from utils import logger, find_nearest_index

def extract_profile_data(direction, fixed_position, dicom_handler, mcc_handler=None):
    """
    프로파일 데이터 추출 함수
    
    Args:
        direction: "vertical" 또는 "horizontal"
        fixed_position: 고정된 위치(mm)
        dicom_handler: DICOM 영상 및 좌표 정보가 포함된 핸들러 객체
        mcc_handler: MCC 영상 및 좌표 정보가 포함된 핸들러 객체(선택 사항)
    
    Returns:
        프로파일 데이터 포함 딕셔너리
    """
    # 프로파일 데이터 저장용 딕셔너리 초기화
    profile_data = {'type': direction, 'fixed_pos': fixed_position}
    
    # DICOM 이미지 데이터 추출
    dicom_image = dicom_handler.get_pixel_data()
    dicom_phys_x_mesh = dicom_handler.phys_x_mesh
    dicom_phys_y_mesh = dicom_handler.phys_y_mesh
    
    try:
        # 프로파일 방향에 따라 다르게 처리
        if direction == "vertical":
            # 수직 프로파일: x 좌표 고정, y 변화
            phys_x = fixed_position  # 고정된 물리적 x 좌표(mm)
            
            # 물리적 좌표에서 가장 가까운 x 인덱스 찾기
            phys_x_coords = dicom_phys_x_mesh[0, :]
            closest_x_idx = find_nearest_index(phys_x_coords, phys_x)
            
            # 해당 열의 물리적 y 좌표 가져오기
            phys_y_coords = dicom_phys_y_mesh[:, closest_x_idx]
            
            # DICOM 값 추출
            dicom_values = dicom_image[:, closest_x_idx]
            
            # 데이터 저장
            profile_data['phys_coords'] = phys_y_coords  # 물리적 y 좌표(mm)
            profile_data['dicom_values'] = dicom_values
            
            # MCC 데이터가 있는 경우 처리
            if mcc_handler is not None:
                mcc_image = mcc_handler.get_matrix_data()
                mcc_phys_x_mesh = mcc_handler.phys_x_mesh
                mcc_phys_y_mesh = mcc_handler.phys_y_mesh
                
                # 물리적 좌표에서 가장 가까운 x 인덱스 찾기
                mcc_phys_x_array = mcc_phys_x_mesh[0, :]
                mcc_closest_x_idx = find_nearest_index(mcc_phys_x_array, phys_x)
                
                # 유효한 MCC 값만 추출(-1 이상인 값)
                valid_indices = np.where(mcc_image[:, mcc_closest_x_idx] >= 0)[0]
                
                if len(valid_indices) > 0:
                    # 유효한 해상도에서 실제 MCC 데이터 포인트 가져오기
                    mcc_phys_y_coords = mcc_phys_y_mesh[valid_indices, mcc_closest_x_idx]
                    mcc_values = mcc_image[valid_indices, mcc_closest_x_idx]
                    
                    # MCC 위치에서의 DICOM 값을 위한 배열 생성
                    dicom_at_mcc_positions = np.full_like(mcc_values, np.nan)
                    
                    # 각 MCC 포인트에 대해 가장 가까운 DICOM 값 찾기
                    for i, mcc_y in enumerate(mcc_phys_y_coords):
                        closest_y_idx = find_nearest_index(phys_y_coords, mcc_y)
                        dicom_at_mcc_positions[i] = dicom_values[closest_y_idx]
                    
                    # 전체 프로파일 시각화를 위한 보간 생성
                    if len(mcc_values) > 1:
                        mcc_interp = np.interp(
                            phys_y_coords,
                            mcc_phys_y_coords,
                            mcc_values,
                            left=np.nan, right=np.nan
                        )
                        
                        profile_data['mcc_phys_coords'] = mcc_phys_y_coords
                        profile_data['mcc_values'] = mcc_values
                        profile_data['mcc_interp'] = mcc_interp
                        
                        # 테이블용 MCC 위치에서의 DICOM 값 저장
                        profile_data['dicom_at_mcc'] = dicom_at_mcc_positions
                        
        else:
            # 수평 프로파일: y 좌표 고정, x 변화
            phys_y = fixed_position  # 고정된 물리적 y 좌표(mm)
            
            # 물리적 좌표에서 가장 가까운 y 인덱스 찾기
            phys_y_coords = dicom_phys_y_mesh[:, 0]
            closest_y_idx = find_nearest_index(phys_y_coords, phys_y)
            
            # 해당 행의 물리적 x 좌표 가져오기
            phys_x_coords = dicom_phys_x_mesh[closest_y_idx, :]
            
            # DICOM 값 추출
            dicom_values = dicom_image[closest_y_idx, :]
            
            # 데이터 저장
            profile_data['phys_coords'] = phys_x_coords  # 물리적 x 좌표(mm)
            profile_data['dicom_values'] = dicom_values
            
            # MCC 데이터가 있는 경우 처리
            if mcc_handler is not None:
                mcc_image = mcc_handler.get_matrix_data()
                mcc_phys_x_mesh = mcc_handler.phys_x_mesh
                mcc_phys_y_mesh = mcc_handler.phys_y_mesh
                
                # 물리적 좌표에서 가장 가까운 y 인덱스 찾기
                mcc_phys_y_array = mcc_phys_y_mesh[:, 0]
                mcc_closest_y_idx = find_nearest_index(mcc_phys_y_array, phys_y)
                
                # 유효한 MCC 값만 추출(-1 이상인 값)
                valid_indices = np.where(mcc_image[mcc_closest_y_idx, :] >= 0)[0]
                
                if len(valid_indices) > 0:
                    # 유효한 해상도에서 실제 MCC 데이터 포인트 가져오기
                    mcc_phys_x_coords = mcc_phys_x_mesh[mcc_closest_y_idx, valid_indices]
                    mcc_values = mcc_image[mcc_closest_y_idx, valid_indices]
                    
                    # MCC 위치에서의 DICOM 값을 위한 배열 생성
                    dicom_at_mcc_positions = np.full_like(mcc_values, np.nan)
                    
                    # 각 MCC 포인트에 대해 가장 가까운 DICOM 값 찾기
                    for i, mcc_x in enumerate(mcc_phys_x_coords):
                        closest_x_idx = find_nearest_index(phys_x_coords, mcc_x)
                        dicom_at_mcc_positions[i] = dicom_values[closest_x_idx]
                    
                    # 전체 프로파일 시각화를 위한 보간 생성
                    if len(mcc_values) > 1:
                        mcc_interp = np.interp(
                            phys_x_coords,
                            mcc_phys_x_coords,
                            mcc_values,
                            left=np.nan, right=np.nan
                        )
                        
                        profile_data['mcc_phys_coords'] = mcc_phys_x_coords
                        profile_data['mcc_values'] = mcc_values
                        profile_data['mcc_interp'] = mcc_interp
                        
                        # 테이블용 MCC 위치에서의 DICOM 값 저장
                        profile_data['dicom_at_mcc'] = dicom_at_mcc_positions
    
        return profile_data
        
    except Exception as e:
        logger.error(f"프로파일 데이터 추출 오류: {str(e)}")
        # 오류 발생 시 기본 데이터만 반환
        return profile_data


def perform_gamma_analysis(reference_handler, evaluation_handler, 
                          distance_mm_threshold, dose_percent_threshold, 
                          global_normalisation=True, threshold=10, max_gamma=3.0):
    """
    감마 분석 수행 함수
    
    Args:
        reference_handler: 참조 데이터 핸들러(MCC)
        evaluation_handler: 평가 데이터 핸들러(DICOM)
        distance_mm_threshold: DTA 임계값(mm)
        dose_percent_threshold: DD 임계값(%)
        global_normalisation: 전역 정규화 사용 여부
        threshold: 최대값의 %로 표현되는 임계값
        max_gamma: 최대 감마 값
        
    Returns:
        (gamma_map, 통계 정보 딕셔너리, 물리적 범위)
    """
    try:
        # 데이터 준비 - 역할 변경: MCC를 참조로, DICOM을 평가로 사용
        reference = reference_handler.get_matrix_data().copy()  # MCC를 참조로 사용
        evaluation = evaluation_handler.get_pixel_data().copy()  # DICOM을 평가로 사용
        
        # MCC 물리적 차원 가져오기
        mcc_spacing_x, mcc_spacing_y = reference_handler.get_spacing()
        mcc_height, mcc_width = reference.shape
        
        # DICOM 데이터 차원
        dicom_height, dicom_width = evaluation.shape
        
        # MCC 그리드에 정렬된 참조 데이터 생성
        aligned_reference = np.full((mcc_height, mcc_width), np.nan)
        # 유효한 MCC 값만 포함(-1 이상)
        aligned_reference = np.where(reference >= 0, reference, np.nan)
        
        # DICOM 데이터를 MCC 그리드에 맞춰 정렬
        aligned_evaluation = np.full_like(aligned_reference, np.nan)
        
        # 벡터화된 접근법으로 중첩 루프 제거
        # 물리적 좌표 메쉬를 이용하여 MCC와 DICOM 간 매핑 수행
        for i in range(mcc_height):
            for j in range(mcc_width):
                if np.isnan(aligned_reference[i, j]):
                    continue  # 참조 데이터가 없는 위치는 건너뜀
                
                # MCC 인덱스를 물리적 좌표로 변환
                phys_x, phys_y = reference_handler.pixel_to_physical_coord(j, i)
                
                # 물리적 좌표를 DICOM 인덱스로 변환
                dicom_x, dicom_y = evaluation_handler.physical_to_pixel_coord(phys_x, phys_y)
                
                # DICOM 그리드 범위 내에 있는지 확인
                if 0 <= dicom_y < dicom_height and 0 <= dicom_x < dicom_width:
                    aligned_evaluation[i, j] = evaluation[dicom_y, dicom_x]
        
        # 임계값 적용
        ref_max = np.nanmax(aligned_reference)
        ref_threshold = threshold / 100 * ref_max
        
        # 유효 마스크 생성
        ref_mask = aligned_reference > ref_threshold
        eval_mask = ~np.isnan(aligned_evaluation)
        valid_mask = ref_mask & eval_mask
        
        # 마스크된 배열 생성
        masked_reference = np.copy(aligned_reference)
        masked_reference[~valid_mask] = 0
        
        masked_evaluation = np.copy(aligned_evaluation)
        masked_evaluation[~valid_mask] = 0
        
        # MCC 그리드용 좌표 생성(참조 데이터 그리드)
        phys_extent = reference_handler.get_physical_extent()
        x_coords = np.linspace(phys_extent[0], phys_extent[1], mcc_width)
        y_coords = np.linspace(phys_extent[2], phys_extent[3], mcc_height)
        
        # 좌표와 선량 데이터 준비
        coords = (y_coords, x_coords)  # pymedphys는 (y, x) 순서로 좌표를 받음
        reference_dose = masked_reference
        evaluation_dose = masked_evaluation
        
        # pymedphys.gamma 함수 사용하여 감마 분석 수행
        gamma_map = pymedphys.gamma(
            coords, reference_dose,
            coords, evaluation_dose,
            dose_percent_threshold=dose_percent_threshold,
            distance_mm_threshold=distance_mm_threshold,
            lower_percent_dose_cutoff=threshold,
            global_normalisation=global_normalisation,
            max_gamma=max_gamma,
            random_subset=None  # 전체 포인트 사용
        )
        
        # NaN 값 마스킹
        gamma_map[~valid_mask] = np.nan
        
        # 감마 통계 계산
        valid_gamma = ~np.isnan(gamma_map)
        gamma_values = gamma_map[valid_gamma]
        
        gamma_stats = {}
        if len(gamma_values) > 0:
            gamma_stats['pass_rate'] = 100 * np.sum(gamma_values < 1.0) / len(gamma_values)
            gamma_stats['mean'] = np.mean(gamma_values)
            gamma_stats['max'] = np.max(gamma_values)
            gamma_stats['min'] = np.min(gamma_values)
            gamma_stats['mask'] = valid_mask
            
        return gamma_map, gamma_stats, phys_extent
    
    except Exception as e:
        logger.error(f"감마 분석 오류: {str(e)}")
        raise
