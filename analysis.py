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
                          dose_percent_threshold, distance_mm_threshold, 
                          global_normalisation=True, threshold=10, max_gamma=3.0):
    """
    pymedphys를 사용하여 감마 분석을 수행하는 함수.
    MCC 측정 포인트를 참조로, 해당 위치의 DICOM 값을 평가 대상으로 사용합니다.
    
    Args:
        reference_handler: 참조 데이터 핸들러 (MCC)
        evaluation_handler: 평가 데이터 핸들러 (DICOM)
        dose_percent_threshold: DD 임계값 (%)
        distance_mm_threshold: DTA 임계값 (mm)
        global_normalisation: 전역 정규화 사용 여부
        threshold: 최대값의 %로 표현되는 선량 임계값
        max_gamma: 최대 감마 값
        
    Returns:
        (gamma_map, 통계 정보 딕셔너리, 물리적 범위)
    """
    try:
        # 1. 참조 데이터 (MCC) 준비
        mcc_data = reference_handler.get_matrix_data()
        if mcc_data is None:
            raise ValueError("MCC 데이터를 찾을 수 없습니다.")

        # 2. 평가 데이터 (DICOM) 준비
        dicom_data = evaluation_handler.get_pixel_data()
        if dicom_data is None:
            raise ValueError("DICOM 데이터를 찾을 수 없습니다.")

        # 3. 유효한 MCC 포인트에서 참조 및 평가 데이터 추출
        valid_mcc_mask = mcc_data >= 0
        
        # MCC 측정 포인트의 물리적 좌표 추출
        mcc_coords_y = reference_handler.phys_y_mesh[valid_mcc_mask].ravel()
        mcc_coords_x = reference_handler.phys_x_mesh[valid_mcc_mask].ravel()
        dose_reference = mcc_data[valid_mcc_mask].ravel()

        # 4. 동일한 물리적 위치에서 DICOM 값 추출 (보간 사용)
        from scipy.interpolate import griddata
        
        # DICOM 그리드의 물리적 좌표
        dicom_y_coords = evaluation_handler.phys_y_mesh[:, 0]
        dicom_x_coords = evaluation_handler.phys_x_mesh[0, :]
        
        # DICOM 그리드 좌표를 1D 배열로 변환
        dicom_y_flat, dicom_x_flat = np.meshgrid(dicom_y_coords, dicom_x_coords, indexing='ij')
        dicom_y_flat = dicom_y_flat.ravel()
        dicom_x_flat = dicom_x_flat.ravel()
        dicom_dose_flat = dicom_data.ravel()
        
        # MCC 위치에서 DICOM 값 보간
        mcc_points = np.column_stack([mcc_coords_y, mcc_coords_x])
        dicom_points = np.column_stack([dicom_y_flat, dicom_x_flat])
        
        dose_evaluation = griddata(
            dicom_points, dicom_dose_flat, mcc_points, 
            method='linear', fill_value=0.0
        )

        # 5. 1D 감마 분석 수행 (동일한 측정 점에서 비교)
        # 거리 기반 좌표 생성 (원점으로부터의 거리)
        distances = np.sqrt(mcc_coords_y**2 + mcc_coords_x**2)
        
        # 정렬된 인덱스 생성
        sorted_indices = np.argsort(distances)
        
        axes_reference = distances[sorted_indices]
        dose_ref_sorted = dose_reference[sorted_indices]
        dose_eval_sorted = dose_evaluation[sorted_indices]

        # 6. 전역 정규화 값 설정
        if global_normalisation:
            global_norm_value = np.max(dose_ref_sorted)
        else:
            global_norm_value = None

        # 7. 감마 분석 수행
        gamma_values_1d = pymedphys.gamma(
            axes_reference, dose_ref_sorted,
            axes_reference, dose_eval_sorted,  # 같은 위치에서 비교
            dose_percent_threshold,
            distance_mm_threshold,
            lower_percent_dose_cutoff=threshold,
            local_gamma=not global_normalisation,
            global_normalisation=global_norm_value,
            max_gamma=max_gamma
        )
        
        # 원래 순서로 복원
        gamma_values = np.empty_like(gamma_values_1d)
        gamma_values[sorted_indices] = gamma_values_1d
        
        # 8. 감마 통계 계산
        gamma_stats = {}
        if len(gamma_values) > 0:
            # NaN 값 제거
            valid_gamma = gamma_values[~np.isnan(gamma_values)]
            if len(valid_gamma) > 0:
                passed = valid_gamma <= 1
                gamma_stats['pass_rate'] = 100 * np.sum(passed) / len(valid_gamma)
                gamma_stats['mean'] = np.mean(valid_gamma)
                gamma_stats['max'] = np.max(valid_gamma)
                gamma_stats['min'] = np.min(valid_gamma)
                gamma_stats['total_points'] = len(valid_gamma)
            else:
                gamma_stats['pass_rate'] = 0
                gamma_stats['mean'] = 0
                gamma_stats['max'] = 0
                gamma_stats['min'] = 0
                gamma_stats['total_points'] = 0

        # 9. 시각화를 위한 2D 감마 맵 생성
        gamma_map = np.full_like(mcc_data, np.nan, dtype=float)
        gamma_map[valid_mcc_mask] = gamma_values
        
        phys_extent = reference_handler.get_physical_extent()
        
        logger.info(f"감마 분석 완료: {len(dose_reference)}개 측정점에서 통과율 {gamma_stats.get('pass_rate', 0):.1f}%")
            
        return gamma_map, gamma_stats, phys_extent
    
    except Exception as e:
        logger.error(f"감마 분석 오류: {str(e)}")
        raise