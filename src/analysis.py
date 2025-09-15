
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

import pymedphys
from scipy.interpolate import griddata

def perform_gamma_analysis(reference_handler, evaluation_handler,
                           dose_percent_threshold, distance_mm_threshold,
                           global_normalisation=True, threshold=10, max_gamma=None):
    """
    Perform gamma analysis using pymedphys.gamma.

    Since the reference data (MCC) is sparse and the evaluation data (DICOM) is a grid,
    the reference data is first interpolated onto the evaluation grid. Then, the two grids
    are compared using the gamma index.
    """
    try:
        # Step 1: Extract reference data (from MCC file)
        # This is the measured data, which is considered the ground truth.
        mcc_dose_data = reference_handler.get_matrix_data()
        if mcc_dose_data is None:
            raise ValueError("MCC data not found.")

        valid_indices = np.where(mcc_dose_data >= 0)
        mcc_coords_phys = np.array([reference_handler.pixel_to_physical_coord(px, py) for py, px in np.vstack(valid_indices).T])
        mcc_dose_values = mcc_dose_data[valid_indices]
        phys_extent = reference_handler.get_physical_extent()

        if mcc_dose_values.size == 0:
            raise ValueError("No valid measurement data in MCC file.")

        # Step 2: Extract evaluation data (from DICOM file)
        # This is the calculated dose from the treatment planning system.
        dicom_dose_grid = evaluation_handler.get_pixel_data()
        if dicom_dose_grid is None:
            raise ValueError("DICOM data not found.")

        dicom_phys_x_mesh, dicom_phys_y_mesh = evaluation_handler.phys_x_mesh, evaluation_handler.phys_y_mesh
        
        # The axes for the evaluation grid, used for both interpolation and gamma analysis.
        axes_eval_grid = (dicom_phys_y_mesh[:, 0], dicom_phys_x_mesh[0, :])

        # Step 3: Align the two datasets based on the maximum dose point
        # This is a common method to align measured and calculated dose distributions.
        max_dose_idx_mcc = np.argmax(mcc_dose_values)
        max_dose_coord_mcc = mcc_coords_phys[max_dose_idx_mcc]

        max_dose_pixel_idx_dicom = np.unravel_index(np.argmax(dicom_dose_grid), dicom_dose_grid.shape)
        max_dose_coord_dicom = evaluation_handler.pixel_to_physical_coord(max_dose_pixel_idx_dicom[1], max_dose_pixel_idx_dicom[0])

        shift = np.array(max_dose_coord_dicom) - np.array(max_dose_coord_mcc)
        mcc_coords_phys_shifted = mcc_coords_phys + shift
        logger.info(f"Coordinate alignment complete (shift: dx={shift[0]:.2f}, dy={shift[1]:.2f} mm)")

        # Step 4: Interpolate the sparse reference (MCC) data onto the evaluation (DICOM) grid
        # pymedphys.gamma requires both datasets to be on grids.
        dose_ref_gridded = griddata(
            mcc_coords_phys_shifted,
            mcc_dose_values,
            (dicom_phys_x_mesh, dicom_phys_y_mesh),
            method='linear',
            fill_value=0
        )

        # Step 5: Perform gamma analysis using pymedphys
        # Now that both reference and evaluation data are on the same grid, we can compare them.
        gamma_grid = pymedphys.gamma(
            axes_reference=axes_eval_grid,
            dose_reference=dose_ref_gridded,
            axes_evaluation=axes_eval_grid,
            dose_evaluation=dicom_dose_grid,
            dose_percent_threshold=dose_percent_threshold,
            distance_mm_threshold=distance_mm_threshold,
            lower_percent_dose_cutoff=threshold,
            global_normalisation=np.max(mcc_dose_values) if global_normalisation else None,
            interp_algo='scipy',
        )

        # Step 6: Extract gamma values at the original MCC measurement points and calculate statistics
        # We use 'nearest' interpolation to find the gamma value on the grid closest to each original measurement point.
        gamma_values_at_mcc_points = griddata(
            (dicom_phys_x_mesh.ravel(), dicom_phys_y_mesh.ravel()),
            gamma_grid.ravel(),
            mcc_coords_phys_shifted,
            method='nearest'
        )

        gamma_map_for_display = np.full_like(mcc_dose_data, np.nan)
        gamma_map_for_display[valid_indices] = gamma_values_at_mcc_points

        gamma_stats = {}
        valid_gamma = gamma_values_at_mcc_points[~np.isnan(gamma_values_at_mcc_points)]

        if len(valid_gamma) > 0:
            passed = valid_gamma <= 1
            gamma_stats['pass_rate'] = 100 * np.sum(passed) / len(valid_gamma)
            gamma_stats['mean'] = np.mean(valid_gamma)
            gamma_stats['max'] = np.max(valid_gamma)
            gamma_stats['min'] = np.min(valid_gamma)
            gamma_stats['total_points'] = len(valid_gamma)
        else:
            gamma_stats.update({'pass_rate': 0, 'mean': 0, 'max': 0, 'min': 0, 'total_points': 0})

        logger.info(f"Gamma analysis complete: {gamma_stats.get('total_points', 0)} points analyzed, pass rate {gamma_stats.get('pass_rate', 0):.1f}%")

        return gamma_map_for_display, gamma_stats, phys_extent

    except Exception as e:
        logger.error(f"Error during gamma analysis: {str(e)}")
        raise
