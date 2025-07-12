
import numpy as np
from scipy.interpolate import interpn
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
                          global_normalisation=True, threshold=10, max_gamma=None):
    """
    수동 DTA 검색 방식을 사용하여 감마 분석을 수행합니다.
    - 기준: MCC의 각 유효 측정점
    - 대상: DICOM의 조밀한 그리드
    - 방식: 각 MCC 점을 기준으로 DTA 반경 내 모든 DICOM 점을 검색하여
            선량 차이가 최소인 점을 찾아 감마를 계산합니다.
    """
    try:
        # 1. 데이터 준비 및 좌표 정렬
        # 1-1. MCC 유효 측정점 추출
        mcc_dose_data = reference_handler.get_matrix_data()
        if mcc_dose_data is None:
            raise ValueError("MCC 데이터를 찾을 수 없습니다.")
            
        valid_indices = np.where(mcc_dose_data >= 0)
        coords_ref_phys = np.array([reference_handler.pixel_to_physical_coord(px, py) for py, px in np.vstack(valid_indices).T])
        dose_ref_points = mcc_dose_data[valid_indices]
        phys_extent = reference_handler.get_physical_extent()

        if dose_ref_points.size == 0:
            raise ValueError("MCC 파일에 유효한 측정 데이터가 없습니다.")

        # 1-2. DICOM 데이터 준비
        dicom_dose_data = evaluation_handler.get_pixel_data()
        if dicom_dose_data is None:
            raise ValueError("DICOM 데이터를 찾을 수 없습니다.")

        # 1-3. 최대 선량점 기준 좌표 정렬
        max_dose_idx_mcc = np.argmax(dose_ref_points)
        max_dose_phys_coord_mcc = coords_ref_phys[max_dose_idx_mcc]
        
        max_dose_pixel_idx_dicom = np.unravel_index(np.argmax(dicom_dose_data), dicom_dose_data.shape)
        max_dose_phys_coord_dicom = evaluation_handler.pixel_to_physical_coord(max_dose_pixel_idx_dicom[1], max_dose_pixel_idx_dicom[0])

        shift = np.array(max_dose_phys_coord_dicom) - np.array(max_dose_phys_coord_mcc)
        coords_ref_phys_shifted = coords_ref_phys + shift
        logger.info(f"좌표 정렬 완료 (이동량: dx={shift[0]:.2f}, dy={shift[1]:.2f} mm)")

        # 2. 감마 계산
        gamma_values = []
        max_ref_dose = np.max(dose_ref_points)
        dd_abs_threshold = (dose_percent_threshold / 100.0) * max_ref_dose
        cutoff_value = (threshold / 100.0) * max_ref_dose

        points_analyzed = 0
        gamma_map_for_display = np.full_like(mcc_dose_data, np.nan)

        for i in range(len(coords_ref_phys_shifted)):
            coord_ref = coords_ref_phys_shifted[i]
            dose_ref = dose_ref_points[i]

            if dose_ref < cutoff_value:
                continue
            points_analyzed += 1

            min_x_search, min_y_search = evaluation_handler.physical_to_pixel_coord(coord_ref[0] - distance_mm_threshold, coord_ref[1] - distance_mm_threshold)
            max_x_search, max_y_search = evaluation_handler.physical_to_pixel_coord(coord_ref[0] + distance_mm_threshold, coord_ref[1] + distance_mm_threshold)
            
            candidate_pixels = []
            for r in range(min_y_search, max_y_search + 1):
                for c in range(min_x_search, max_x_search + 1):
                    try:
                        phys_coord_cand = evaluation_handler.pixel_to_physical_coord(c, r)
                        dist_sq = (phys_coord_cand[0] - coord_ref[0])**2 + (phys_coord_cand[1] - coord_ref[1])**2
                        
                        if dist_sq <= distance_mm_threshold**2:
                            dose_cand = dicom_dose_data[r, c]
                            candidate_pixels.append({'dist_sq': dist_sq, 'dose_diff': abs(dose_cand - dose_ref), 'dist': np.sqrt(dist_sq)})
                    except IndexError:
                        continue

            if not candidate_pixels:
                gamma_val = np.inf
            else:
                best_candidate = min(candidate_pixels, key=lambda x: x['dose_diff'])
                gamma_sq = (best_candidate['dist'] / distance_mm_threshold)**2 + (best_candidate['dose_diff'] / dd_abs_threshold)**2
                gamma_val = np.sqrt(gamma_sq)
            
            # Store gamma value for stats
            gamma_values.append(gamma_val)
            # Store gamma value in a 2D map for display
            gamma_map_for_display[valid_indices[0][i], valid_indices[1][i]] = gamma_val

        # 3. 감마 통계 계산
        gamma_stats = {}
        valid_gamma = np.array(gamma_values)
        valid_gamma = valid_gamma[~np.isinf(valid_gamma)]

        if len(valid_gamma) > 0:
            passed = valid_gamma <= 1
            gamma_stats['pass_rate'] = 100 * np.sum(passed) / len(valid_gamma)
            gamma_stats['mean'] = np.mean(valid_gamma)
            gamma_stats['max'] = np.max(valid_gamma)
            gamma_stats['min'] = np.min(valid_gamma)
            gamma_stats['total_points'] = len(valid_gamma)
        else:
            gamma_stats.update({'pass_rate': 0, 'mean': 0, 'max': 0, 'min': 0, 'total_points': 0})
        
        logger.info(f"감마 분석 완료: {gamma_stats.get('total_points', 0)}개 지점에서 통과율 {gamma_stats.get('pass_rate', 0):.1f}%")
            
        return gamma_map_for_display, gamma_stats, phys_extent
    
    except Exception as e:
        logger.error(f"감마 분석 오류: {str(e)}")
        raise
