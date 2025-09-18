
"""
This module provides functions for analyzing and comparing DICOM and MCC data,
including profile extraction and gamma analysis.
"""
import numpy as np
from scipy.interpolate import griddata
from utils import logger, find_nearest_index, save_map_to_csv
import os

def extract_profile_data(direction, fixed_position, dicom_handler, mcc_handler=None):
    """
    Extracts profile data from DICOM and MCC datasets.

    Args:
        direction (str): "vertical" or "horizontal".
        fixed_position (float): The fixed position in mm.
        dicom_handler: Handler object containing DICOM image and coordinate information.
        mcc_handler: Handler object containing MCC image and coordinate information (optional).

    Returns:
        dict: A dictionary containing the profile data.
    """
    profile_data = {'type': direction, 'fixed_pos': fixed_position}
    
    dicom_image = dicom_handler.get_pixel_data()
    if dicom_image is None:
        logger.warning("DICOM pixel data not available for profile extraction.")
        return profile_data
    
    try:
        # Define axis-dependent variables based on direction
        if direction == "vertical":
            # Vertical profile: x is fixed, y is the profile axis
            fixed_axis_coords_dicom = dicom_handler.phys_x_mesh[0, :]
            profile_axis_mesh_dicom = dicom_handler.phys_y_mesh
            slicer_dicom = lambda idx: (slice(None), idx)
            if mcc_handler:
                mcc_fixed_axis_coords = mcc_handler.phys_x_mesh[0, :]
                mcc_profile_axis_mesh = mcc_handler.phys_y_mesh
                slicer_mcc = lambda idx: (slice(None), idx)
                sort_required = True  # MCC y-axis is inverted and needs sorting for interpolation
        else:  # "horizontal"
            # Horizontal profile: y is fixed, x is the profile axis
            fixed_axis_coords_dicom = dicom_handler.phys_y_mesh[:, 0]
            profile_axis_mesh_dicom = dicom_handler.phys_x_mesh
            slicer_dicom = lambda idx: (idx, slice(None))
            if mcc_handler:
                mcc_fixed_axis_coords = mcc_handler.phys_y_mesh[:, 0]
                mcc_profile_axis_mesh = mcc_handler.phys_x_mesh
                slicer_mcc = lambda idx: (idx, slice(None))
                sort_required = False # MCC x-axis is already sorted

        # --- Common Logic for both directions ---

        # 1. Extract DICOM profile
        closest_idx_dicom = find_nearest_index(fixed_axis_coords_dicom, fixed_position)
        profile_coords_dicom = profile_axis_mesh_dicom[slicer_dicom(closest_idx_dicom)]
        dicom_values = dicom_image[slicer_dicom(closest_idx_dicom)]

        # For vertical profiles, the physical y-coordinate is descending.
        # To ensure consistency for interpolation and plotting, we make it ascending.
        if sort_required:
            # Reverse both arrays to sort them in ascending order of coordinates.
            profile_coords_dicom = profile_coords_dicom[::-1]
            dicom_values = dicom_values[::-1]

        profile_data['phys_coords'] = profile_coords_dicom
        profile_data['dicom_values'] = dicom_values

        # 2. Process MCC data if available
        if mcc_handler and mcc_handler.get_matrix_data() is not None:
            mcc_image = mcc_handler.get_matrix_data()
            
            closest_idx_mcc = find_nearest_index(mcc_fixed_axis_coords, fixed_position)
            mcc_line_values = mcc_image[slicer_mcc(closest_idx_mcc)]
            
            valid_indices = np.where(mcc_line_values >= 0)[0]
            
            if len(valid_indices) > 0:
                # Get physical coordinates and dose values for valid MCC points
                if direction == "vertical":
                    mcc_phys_coords = mcc_profile_axis_mesh[valid_indices, closest_idx_mcc]
                else: # horizontal
                    mcc_phys_coords = mcc_profile_axis_mesh[closest_idx_mcc, valid_indices]
                
                mcc_values = mcc_line_values[valid_indices]

                # For vertical profiles, the physical y-coordinate is descending.
                # We sort it here to ensure all subsequent operations use ascending coordinates.
                if sort_required:
                    mcc_phys_coords = mcc_phys_coords[::-1]
                    mcc_values = mcc_values[::-1]
                
                # Find corresponding DICOM values at MCC measurement points
                dicom_at_mcc_positions = np.array([
                    dicom_values[find_nearest_index(profile_coords_dicom, pos)] for pos in mcc_phys_coords
                ])
                
                # Interpolate MCC data for smooth plotting
                if len(mcc_values) > 1:
                    # The coordinate arrays are now guaranteed to be ascending,
                    # so we can interpolate directly.
                    mcc_interp = np.interp(
                        profile_coords_dicom,
                        mcc_phys_coords,
                        mcc_values,
                        left=np.nan, right=np.nan
                    )
                    profile_data['mcc_interp'] = mcc_interp

                profile_data['mcc_phys_coords'] = mcc_phys_coords
                profile_data['mcc_values'] = mcc_values
                profile_data['dicom_at_mcc'] = dicom_at_mcc_positions

        return profile_data
        
    except Exception as e:
        logger.error(f"프로파일 데이터 추출 오류: {str(e)}", exc_info=True)
        return profile_data

def perform_gamma_analysis(reference_handler, evaluation_handler,
                           dose_percent_threshold, distance_mm_threshold,
                           global_normalisation=True, threshold=10, max_gamma=None,
                           save_csv=False, csv_dir=None):
    """
    Performs gamma index analysis directly between sparse reference points (MCC)
    and a dense evaluation grid (DICOM) without interpolation.

    Args:
        reference_handler: Handler for the reference data (MCC).
        evaluation_handler: Handler for the evaluation data (DICOM).
        dose_percent_threshold (float): Dose difference criterion (%).
        distance_mm_threshold (float): Distance-to-agreement criterion (mm).
        global_normalisation (bool): Whether to use global normalization.
        threshold (int): The lower dose threshold for including points in the analysis (%).
        save_csv (bool): Whether to save the analysis maps to CSV files.
        csv_dir (str): The directory to save the CSV files in.

    Returns:
        tuple: A tuple containing:
            - gamma_map (np.ndarray): The gamma map.
            - gamma_stats (dict): Statistics of the gamma analysis.
            - phys_extent (list): The physical extent of the analysis.
            - mcc_interp_data (np.ndarray): Interpolated MCC data for visualization.
            - dd_map (np.ndarray): The dose difference map.
            - dta_map (np.ndarray): The distance-to-agreement map.
            - dd_stats (dict): Statistics of the dose difference analysis.
            - dta_stats (dict): Statistics of the distance-to-agreement analysis.
    """
    try:
        # --- Step 1: Extract and filter reference data (MCC) ---
        mcc_dose_data = reference_handler.get_matrix_data()
        if mcc_dose_data is None:
            raise ValueError("MCC data not found in reference_handler.")

        # Get all valid (>=0) measurement points and their coordinates
        all_valid_indices = np.where(mcc_dose_data >= 0)
        if all_valid_indices[0].size == 0:
            raise ValueError("No valid measurement data (>= 0) in MCC file.")

        # Vectorized calculation of physical coordinates from pixel coordinates
        y_pix, x_pix = all_valid_indices
        handler = reference_handler
        full_grid_px = x_pix + handler.crop_pixel_offset[0]
        full_grid_py = y_pix + handler.crop_pixel_offset[1]
        phys_x_all = (full_grid_px - handler.mcc_origin_x) * handler.mcc_spacing_x
        phys_y_all = (full_grid_py - handler.mcc_origin_y) * handler.mcc_spacing_y
        all_mcc_coords_phys = np.vstack((phys_x_all, phys_y_all)).T

        all_mcc_dose_values = mcc_dose_data[all_valid_indices]

        # Determine normalization dose for thresholding
        if global_normalisation:
            norm_dose = np.max(all_mcc_dose_values)
        else:
            norm_dose = 1.0  # Local normalization handled per point
        
        if norm_dose == 0:
            raise ValueError("Cannot determine normalization dose (max reference dose is zero).")

        # Filter points based on the lower dose cutoff threshold
        threshold_dose = (threshold / 100.0) * norm_dose
        analysis_mask = all_mcc_dose_values >= threshold_dose

        # --- Step 2: Extract evaluation data (DICOM) ---
        dicom_dose_grid = evaluation_handler.get_pixel_data()
        if dicom_dose_grid is None:
            raise ValueError("DICOM data not found in evaluation_handler.")

        dicom_phys_x_mesh, dicom_phys_y_mesh = evaluation_handler.phys_x_mesh, evaluation_handler.phys_y_mesh
        phys_x_dicom = dicom_phys_x_mesh[0, :]  # x coordinates
        phys_y_dicom = dicom_phys_y_mesh[:, 0]  # y coordinates
        phys_extent = reference_handler.get_physical_extent()

        # If no points are left after filtering, return early.
        if not np.any(analysis_mask):
            logger.warning(f"No MCC data points above the {threshold}% dose threshold ({threshold_dose:.2f} Gy). Gamma analysis will be skipped.")
            gamma_stats = {'pass_rate': 100, 'mean': 0, 'max': 0, 'min': 0, 'total_points': 0}
            gamma_map_for_display = np.full_like(mcc_dose_data, np.nan)
            # Still create interpolated data for the report
            mcc_interp_data = griddata(
                all_mcc_coords_phys, all_mcc_dose_values,
                (dicom_phys_x_mesh, dicom_phys_y_mesh),
                method='linear', fill_value=0
            )
            # Return empty dd and dta maps when no analysis is performed
            dd_map_empty = np.full_like(mcc_dose_data, np.nan)
            dta_map_empty = np.full_like(mcc_dose_data, np.nan)
            dd_stats_empty = {'mean': 0, 'max': 0, 'min': 0, 'std': 0, 'total_points': 0}
            dta_stats_empty = {'mean': 0, 'max': 0, 'min': 0, 'std': 0, 'total_points': 0}
            return gamma_map_for_display, gamma_stats, phys_extent, mcc_interp_data, dd_map_empty, dta_map_empty, dd_stats_empty, dta_stats_empty

        # These are the points that will be used in the gamma calculation
        mcc_dose_for_gamma = all_mcc_dose_values[analysis_mask]
        points_ref = all_mcc_coords_phys[analysis_mask]
        doses_ref = mcc_dose_for_gamma

        # Get the original pixel indices of the analyzed points to place results back into the map
        original_indices_for_gamma = (all_valid_indices[0][analysis_mask], all_valid_indices[1][analysis_mask])

        # --- Step 3: Perform manual gamma calculation ---
        # Setup criteria
        dta_criteria_sq = distance_mm_threshold ** 2

        # Initialize gamma, dd, and dta values arrays
        gamma_values = np.full(len(points_ref), np.inf)
        dd_values = np.full(len(points_ref), np.inf)
        dta_values = np.full(len(points_ref), np.inf)

        logger.info(f"Starting manual gamma calculation for {len(points_ref)} reference points...")

        # 각 기준(MCC) 포인트에 대해 감마 계산
        for i, (point_ref, dose_ref) in enumerate(zip(points_ref, doses_ref)):
            # Dose difference criteria (global or local normalization)
            if global_normalisation:
                dd_criteria_sq = (dose_percent_threshold / 100.0 * norm_dose) ** 2
            else:
                dd_criteria_sq = (dose_percent_threshold / 100.0 * dose_ref) ** 2

            # 1. 거리 계산: 현재 기준점과 모든 DICOM 픽셀 간의 거리
            # 최적화: DTA 내의 점들만 고려
            search_radius = distance_mm_threshold * 2  # 탐색 반경 (DTA의 2배 정도)

            # DICOM 좌표에서 탐색 영역 필터링
            min_x, max_x = point_ref[0] - search_radius, point_ref[0] + search_radius
            min_y, max_y = point_ref[1] - search_radius, point_ref[1] + search_radius

            x_indices = np.where((phys_x_dicom >= min_x) & (phys_x_dicom <= max_x))[0]
            y_indices = np.where((phys_y_dicom >= min_y) & (phys_y_dicom <= max_y))[0]

            if x_indices.size == 0 or y_indices.size == 0:
                continue

            # 탐색 영역 내의 DICOM 좌표와 선량 값
            eval_coords_y, eval_coords_x = np.meshgrid(phys_y_dicom[y_indices], phys_x_dicom[x_indices], indexing='ij')
            points_eval = np.vstack((eval_coords_x.ravel(), eval_coords_y.ravel())).T
            doses_eval = dicom_dose_grid[np.ix_(y_indices, x_indices)].ravel()

            # 2. 선량 차이 계산
            dose_diff_sq = (doses_eval - dose_ref) ** 2
            
            # 3. 거리 차이 계산
            dist_sq = np.sum((points_eval - point_ref)**2, axis=1)

            # 4. 감마 계산
            gamma_sq = (dist_sq / dta_criteria_sq) + (dose_diff_sq / dd_criteria_sq)

            # 5. 최소 감마 값 및 해당 지점의 dd, dta 저장
            min_idx = np.argmin(gamma_sq)
            min_gamma = np.sqrt(gamma_sq[min_idx])
            gamma_values[i] = min_gamma
            
            # Store dd and dta values at the minimum gamma point
            dd_values[i] = np.sqrt(dose_diff_sq[min_idx]) / (dose_percent_threshold / 100.0 * (norm_dose if global_normalisation else dose_ref))
            dta_values[i] = np.sqrt(dist_sq[min_idx]) / distance_mm_threshold

            # Progress logging for large datasets
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(points_ref)} reference points...")

        # --- Step 4: Create gamma, dd, and dta maps and calculate statistics ---
        gamma_map_for_display = np.full_like(mcc_dose_data, np.nan)
        dd_map_for_display = np.full_like(mcc_dose_data, np.nan)
        dta_map_for_display = np.full_like(mcc_dose_data, np.nan)

        gamma_map_for_display[original_indices_for_gamma] = gamma_values
        dd_map_for_display[original_indices_for_gamma] = dd_values
        dta_map_for_display[original_indices_for_gamma] = dta_values

        gamma_stats = {}
        valid_gamma = gamma_values[~np.isinf(gamma_values) & ~np.isnan(gamma_values)]

        if len(valid_gamma) > 0:
            passed = valid_gamma <= 1
            gamma_stats['pass_rate'] = 100 * np.sum(passed) / len(valid_gamma)
            gamma_stats['mean'] = np.mean(valid_gamma)
            gamma_stats['max'] = np.max(valid_gamma)
            gamma_stats['min'] = np.min(valid_gamma)
            gamma_stats['total_points'] = len(valid_gamma)
        else:
            gamma_stats.update({'pass_rate': 0, 'mean': 0, 'max': 0, 'min': 0, 'total_points': 0})

        # Calculate DD and DTA statistics
        dd_stats = {}
        dta_stats = {}

        valid_dd = dd_values[~np.isinf(dd_values) & ~np.isnan(dd_values)]
        valid_dta = dta_values[~np.isinf(dta_values) & ~np.isnan(dta_values)]
        
        if len(valid_dd) > 0:
            dd_stats['mean'] = np.mean(valid_dd)
            dd_stats['max'] = np.max(valid_dd)
            dd_stats['min'] = np.min(valid_dd)
            dd_stats['std'] = np.std(valid_dd)
            dd_stats['total_points'] = len(valid_dd)
        else:
            dd_stats.update({'mean': 0, 'max': 0, 'min': 0, 'std': 0, 'total_points': 0})
            
        if len(valid_dta) > 0:
            dta_stats['mean'] = np.mean(valid_dta)
            dta_stats['max'] = np.max(valid_dta)
            dta_stats['min'] = np.min(valid_dta)
            dta_stats['std'] = np.std(valid_dta)
            dta_stats['total_points'] = len(valid_dta)
        else:
            dta_stats.update({'mean': 0, 'max': 0, 'min': 0, 'std': 0, 'total_points': 0})

        logger.info(f"Manual gamma analysis complete: {gamma_stats.get('total_points', 0)} points analyzed, pass rate {gamma_stats.get('pass_rate', 0):.1f}%")

        # --- Step 5: Create interpolated MCC data for visualization purposes only ---
        mcc_interp_data = griddata(
            all_mcc_coords_phys,
            all_mcc_dose_values,
            (dicom_phys_x_mesh, dicom_phys_y_mesh),
            method='linear',
            fill_value=0
        )

        # --- Step 6: Save maps to CSV ---
        if save_csv and csv_dir:
            try:
                base_filename = os.path.splitext(os.path.basename(reference_handler.filename))[0]
                mcc_phys_x_mesh = reference_handler.phys_x_mesh
                mcc_phys_y_mesh = reference_handler.phys_y_mesh

                if gamma_stats.get('total_points', 0) > 0:
                    gamma_path = os.path.join(csv_dir, f"{base_filename}_gamma.csv")
                    dd_path = os.path.join(csv_dir, f"{base_filename}_dd.csv")
                    dta_path = os.path.join(csv_dir, f"{base_filename}_dta.csv")
                    save_map_to_csv(gamma_map_for_display, mcc_phys_x_mesh, mcc_phys_y_mesh, gamma_path)
                    save_map_to_csv(dd_map_for_display, mcc_phys_x_mesh, mcc_phys_y_mesh, dd_path)
                    save_map_to_csv(dta_map_for_display, mcc_phys_x_mesh, mcc_phys_y_mesh, dta_path)
            except Exception as e:
                logger.error(f"Failed to save analysis maps to CSV: {e}", exc_info=True)

        return gamma_map_for_display, gamma_stats, phys_extent, mcc_interp_data, dd_map_for_display, dta_map_for_display, dd_stats, dta_stats

    except Exception as e:
        logger.error(f"Error during manual gamma analysis: {str(e)}")
        raise
