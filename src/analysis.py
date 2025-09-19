"""
This module provides functions for analyzing and comparing dose data using
standardized data objects. It includes profile extraction and gamma analysis.
"""
import numpy as np
from scipy.interpolate import griddata, interp1d
from typing import Optional

from .standard_data_model import StandardDoseData
from .utils import logger, find_nearest_index, save_map_to_csv
import os

def extract_profile_data(direction: str, fixed_position: float,
                         dicom_data: StandardDoseData,
                         mcc_data: Optional[StandardDoseData] = None) -> Optional[dict]:
    """
    Extracts profile data from standardized DICOM and MCC data objects.

    Args:
        direction: "vertical" or "horizontal".
        fixed_position: The fixed position in mm.
        dicom_data: The standardized DICOM data object.
        mcc_data: The standardized MCC data object (optional).

    Returns:
        A dictionary containing the profile data, or None if extraction fails.
    """
    profile_data = {'type': direction, 'fixed_pos': fixed_position}
    
    try:
        # 1. Extract DICOM profile (always present)
        if direction == "vertical":
            # Vertical profile: x is fixed, y is the profile axis
            fixed_axis_coords = dicom_data.x_coords
            profile_axis_coords = dicom_data.y_coords
            closest_idx = find_nearest_index(fixed_axis_coords, fixed_position)
            dicom_values = dicom_data.data_grid[:, closest_idx]
        else:  # "horizontal"
            # Horizontal profile: y is fixed, x is the profile axis
            fixed_axis_coords = dicom_data.y_coords
            profile_axis_coords = dicom_data.x_coords
            closest_idx = find_nearest_index(fixed_axis_coords, fixed_position)
            dicom_values = dicom_data.data_grid[closest_idx, :]

        profile_data['phys_coords'] = profile_axis_coords
        profile_data['dicom_values'] = dicom_values

        # 2. Process MCC data if available
        if mcc_data:
            # For smooth plotting, interpolate the dense MCC grid onto the DICOM profile coordinates
            if direction == "vertical":
                # Create an interpolator for the MCC data along the y-axis at the given x
                mcc_interp_func = interp1d(mcc_data.y_coords, mcc_data.data_grid[:, find_nearest_index(mcc_data.x_coords, fixed_position)],
                                           bounds_error=False, fill_value=np.nan)
                profile_data['mcc_interp'] = mcc_interp_func(profile_axis_coords)
            else: # horizontal
                mcc_interp_func = interp1d(mcc_data.x_coords, mcc_data.data_grid[find_nearest_index(mcc_data.y_coords, fixed_position), :],
                                           bounds_error=False, fill_value=np.nan)
                profile_data['mcc_interp'] = mcc_interp_func(profile_axis_coords)

            # For the table, provide the original MCC points and corresponding DICOM values
            if 'original_points' in mcc_data.metadata:
                original_mcc = mcc_data.metadata['original_points']
                profile_data['mcc_phys_coords'] = original_mcc['coords'][:, 0 if direction == 'horizontal' else 1]
                profile_data['mcc_values'] = original_mcc['values']
                
                # Find corresponding DICOM values at these original MCC points
                dicom_interp_func = interp1d(profile_axis_coords, dicom_values, bounds_error=False, fill_value=np.nan)
                profile_data['dicom_at_mcc'] = dicom_interp_func(profile_data['mcc_phys_coords'])

        return profile_data
        
    except Exception as e:
        logger.error(f"Profile data extraction error: {e}", exc_info=True)
        return None

def perform_gamma_analysis(reference_data: StandardDoseData, evaluation_data: StandardDoseData,
                           dose_percent_threshold: float, distance_mm_threshold: float,
                           global_normalisation: bool = True, threshold: int = 10,
                           save_csv: bool = False, csv_dir: Optional[str] = None):
    """
    Performs gamma analysis using sparse reference points (from MCC metadata)
    against a dense evaluation grid (DICOM).

    Args:
        reference_data: Standardized data for reference (MCC), must contain 'original_points' in metadata.
        evaluation_data: Standardized data for evaluation (DICOM).
        dose_percent_threshold: Dose difference criterion (%).
        distance_mm_threshold: DTA criterion (mm).
        global_normalisation: Whether to use global normalization.
        threshold: Lower dose threshold for analysis (%).
        save_csv: Whether to save analysis maps to CSV.
        csv_dir: Directory to save CSV files.

    Returns:
        A tuple containing gamma maps, stats, and other analysis results.
    """
    try:
        # --- Step 1: Extract and filter reference data from metadata ---
        if 'original_points' not in reference_data.metadata:
            raise ValueError("Reference data must contain 'original_points' in metadata for sparse gamma analysis.")
        
        all_mcc_coords_phys = reference_data.metadata['original_points']['coords']
        all_mcc_dose_values = reference_data.metadata['original_points']['values']

        if all_mcc_coords_phys.size == 0:
            raise ValueError("No valid measurement data in reference file.")

        norm_dose = np.max(all_mcc_dose_values) if global_normalisation else 1.0
        if norm_dose == 0:
            raise ValueError("Cannot determine normalization dose (max reference dose is zero).")

        threshold_dose = (threshold / 100.0) * norm_dose
        analysis_mask = all_mcc_dose_values >= threshold_dose

        # --- Step 2: Extract evaluation data ---
        eval_grid = evaluation_data.data_grid
        eval_x_coords = evaluation_data.x_coords
        eval_y_coords = evaluation_data.y_coords
        phys_extent = evaluation_data.physical_extent

        if not np.any(analysis_mask):
            logger.warning(f"No reference points above the {threshold}% dose threshold. Skipping analysis.")
            # Return empty/default values
            empty_stats = {'pass_rate': 100, 'mean': 0, 'max': 0, 'min': 0, 'total_points': 0}
            empty_map = np.full_like(reference_data.data_grid, np.nan)
            return empty_map, empty_stats, phys_extent, reference_data.data_grid, empty_map, empty_map, empty_stats, empty_stats

        points_ref = all_mcc_coords_phys[analysis_mask]
        doses_ref = all_mcc_dose_values[analysis_mask]

        # --- Step 3: Perform gamma calculation ---
        dta_criteria_sq = distance_mm_threshold ** 2
        gamma_values = np.full(len(points_ref), np.inf)
        dd_values = np.full(len(points_ref), np.inf)
        dta_values = np.full(len(points_ref), np.inf)

        logger.info(f"Starting gamma calculation for {len(points_ref)} reference points...")

        for i, (point_ref, dose_ref) in enumerate(zip(points_ref, doses_ref)):
            dd_criteria = (dose_percent_threshold / 100.0) * (norm_dose if global_normalisation else dose_ref)
            if dd_criteria == 0: continue
            dd_criteria_sq = dd_criteria ** 2

            # Search optimization: limit search to a radius around the reference point
            search_radius = distance_mm_threshold * 2.5
            min_x, max_x = point_ref[0] - search_radius, point_ref[0] + search_radius
            min_y, max_y = point_ref[1] - search_radius, point_ref[1] + search_radius

            x_indices = np.where((eval_x_coords >= min_x) & (eval_x_coords <= max_x))[0]
            y_indices = np.where((eval_y_coords >= min_y) & (eval_y_coords <= max_y))[0]

            if x_indices.size == 0 or y_indices.size == 0: continue

            eval_coords_y, eval_coords_x = np.meshgrid(eval_y_coords[y_indices], eval_x_coords[x_indices], indexing='ij')
            points_eval = np.vstack((eval_coords_x.ravel(), eval_coords_y.ravel())).T
            doses_eval = eval_grid[np.ix_(y_indices, x_indices)].ravel()

            dose_diff_sq = (doses_eval - dose_ref) ** 2
            dist_sq = np.sum((points_eval - point_ref)**2, axis=1)
            gamma_sq = (dist_sq / dta_criteria_sq) + (dose_diff_sq / dd_criteria_sq)

            min_idx = np.argmin(gamma_sq)
            gamma_values[i] = np.sqrt(gamma_sq[min_idx])
            dd_values[i] = np.sqrt(dose_diff_sq[min_idx]) / dd_criteria
            dta_values[i] = np.sqrt(dist_sq[min_idx]) / distance_mm_threshold

        # --- Step 4: Create dense maps and calculate statistics ---
        grid_x, grid_y = np.meshgrid(reference_data.x_coords, reference_data.y_coords)

        gamma_map = griddata(points_ref, gamma_values, (grid_x, grid_y), method='cubic', fill_value=np.nan)
        dd_map = griddata(points_ref, dd_values, (grid_x, grid_y), method='cubic', fill_value=np.nan)
        dta_map = griddata(points_ref, dta_values, (grid_x, grid_y), method='cubic', fill_value=np.nan)

        valid_gamma = gamma_values[np.isfinite(gamma_values)]
        gamma_stats = {
            'pass_rate': 100 * np.sum(valid_gamma <= 1) / len(valid_gamma) if len(valid_gamma) > 0 else 100,
            'mean': np.mean(valid_gamma) if len(valid_gamma) > 0 else 0,
            'max': np.max(valid_gamma) if len(valid_gamma) > 0 else 0,
            'min': np.min(valid_gamma) if len(valid_gamma) > 0 else 0,
            'total_points': len(valid_gamma)
        }

        def get_stats(values):
            valid = values[np.isfinite(values)]
            return {
                'mean': np.mean(valid) if len(valid) > 0 else 0, 'max': np.max(valid) if len(valid) > 0 else 0,
                'min': np.min(valid) if len(valid) > 0 else 0, 'std': np.std(valid) if len(valid) > 0 else 0,
                'total_points': len(valid)
            }
        dd_stats = get_stats(dd_values)
        dta_stats = get_stats(dta_values)

        logger.info(f"Gamma analysis complete: {gamma_stats['total_points']} points analyzed, pass rate {gamma_stats['pass_rate']:.1f}%")

        # The interpolated MCC data for visualization is now just the reference data grid
        mcc_interp_data = reference_data.data_grid

        # --- Step 5: Save maps to CSV (Optional) ---
        if save_csv and csv_dir:
            try:
                base_filename = os.path.splitext(os.path.basename(reference_data.metadata.get('filename', 'mcc')))[0]
                if gamma_stats['total_points'] > 0:
                    save_map_to_csv(gamma_map, grid_x, grid_y, os.path.join(csv_dir, f"{base_filename}_gamma.csv"))
                    save_map_to_csv(dd_map, grid_x, grid_y, os.path.join(csv_dir, f"{base_filename}_dd.csv"))
                    save_map_to_csv(dta_map, grid_x, grid_y, os.path.join(csv_dir, f"{base_filename}_dta.csv"))
            except Exception as e:
                logger.error(f"Failed to save analysis maps to CSV: {e}", exc_info=True)

        return gamma_map, gamma_stats, phys_extent, mcc_interp_data, dd_map, dta_map, dd_stats, dta_stats

    except Exception as e:
        logger.error(f"Error during gamma analysis: {e}", exc_info=True)
        raise
