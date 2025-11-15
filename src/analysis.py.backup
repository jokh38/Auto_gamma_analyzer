"""
This module provides functions for analyzing and comparing dose data using
standardized data objects. It includes profile extraction and gamma analysis.
"""
import numpy as np
from scipy.interpolate import griddata, interp1d
from typing import Optional

from .standard_data_model import StandardDoseData, ROI_Data
from .utils import logger, find_nearest_index, save_map_to_csv
import os
 
def extract_profile_data(
    direction: str,
    fixed_position: float,
    dicom_roi: ROI_Data,
    mcc_roi: Optional[ROI_Data] = None
) -> Optional[dict]:
    """
    Extracts profile data from DICOM ROI and corresponding MCC data.

    Args:
        direction: "vertical" or "horizontal".
        fixed_position: The fixed physical position (in mm) of the profile.
        dicom_roi: The DICOM ROI_Data object to extract the base profile from.
        mcc_roi: The optional MCC ROI_Data object for comparison data.

    Returns:
        A dictionary containing the profile data, or None if extraction fails.
    """
    profile_data = {'type': direction, 'fixed_pos': fixed_position}

    try:
        if direction == "vertical":
            # Vertical profile: x is fixed, y is the profile axis
            fixed_axis_coords = dicom_roi.x_coords
            profile_axis_coords = dicom_roi.y_coords
            closest_idx = find_nearest_index(fixed_axis_coords, fixed_position)
            dicom_values = dicom_roi.dose_grid[:, closest_idx]
        else:  # "horizontal"
            # Horizontal profile: y is fixed, x is the profile axis
            fixed_axis_coords = dicom_roi.y_coords
            profile_axis_coords = dicom_roi.x_coords
            closest_idx = find_nearest_index(fixed_axis_coords, fixed_position)
            dicom_values = dicom_roi.dose_grid[closest_idx, :]

        profile_data['phys_coords'] = profile_axis_coords
        profile_data['dicom_values'] = dicom_values

        # 2. Process MCC data if available
        if mcc_roi and 'original_points' in mcc_roi.source_metadata:
            # Get a profile slice from the MCC grid
            if direction == "vertical":
                mcc_fixed_axis = mcc_roi.x_coords
                mcc_profile_axis = mcc_roi.y_coords
                mcc_slice_idx = find_nearest_index(mcc_fixed_axis, fixed_position)
                mcc_slice_values = mcc_roi.dose_grid[:, mcc_slice_idx]
            else: # horizontal
                mcc_fixed_axis = mcc_roi.y_coords
                mcc_profile_axis = mcc_roi.x_coords
                mcc_slice_idx = find_nearest_index(mcc_fixed_axis, fixed_position)
                mcc_slice_values = mcc_roi.dose_grid[mcc_slice_idx, :]

            # Interpolate the 1D MCC slice onto the DICOM profile's coordinates
            interp_func_mcc = interp1d(
                mcc_profile_axis,
                mcc_slice_values,
                bounds_error=False,
                fill_value=0
            )
            profile_data['mcc_interp'] = interp_func_mcc(profile_axis_coords)

            # Find original MCC measurement points near the profile line
            mcc_points_coords = mcc_roi.source_metadata['original_points']['coords']
            mcc_points_values = mcc_roi.source_metadata['original_points']['values']

            spacing = mcc_roi.source_metadata.get('original_spacing', 5.0)
            tolerance = spacing / 2.0

            axis_idx, profile_axis_idx = (0, 1) if direction == "vertical" else (1, 0)

            mask = np.abs(mcc_points_coords[:, axis_idx] - fixed_position) < tolerance

            mcc_coords_on_profile = mcc_points_coords[mask]
            if mcc_coords_on_profile.size > 0:
                profile_data['mcc_phys_coords'] = mcc_coords_on_profile[:, profile_axis_idx]
                profile_data['mcc_values'] = mcc_points_values[mask]

                # Interpolate DICOM dose at original MCC points for comparison table
                interp_func_dicom = interp1d(
                    profile_axis_coords, dicom_values, bounds_error=False, fill_value=np.nan
                )
                profile_data['dicom_at_mcc'] = interp_func_dicom(profile_data['mcc_phys_coords'])

                # Sort MCC data by position for cleaner plotting
                sort_indices = np.argsort(profile_data['mcc_phys_coords'])
                profile_data['mcc_phys_coords'] = profile_data['mcc_phys_coords'][sort_indices]
                profile_data['mcc_values'] = profile_data['mcc_values'][sort_indices]
                profile_data['dicom_at_mcc'] = profile_data['dicom_at_mcc'][sort_indices]

        return profile_data

    except Exception as e:
        logger.error(f"Profile data extraction error: {e}", exc_info=True)
        return None

def perform_gamma_analysis(reference_roi: ROI_Data, evaluation_roi: ROI_Data,
                           dose_percent_threshold: float, distance_mm_threshold: float,
                           global_normalisation: bool = True,
                           save_csv: bool = False, csv_dir: Optional[str] = None):
    """
    Performs gamma analysis using data from ROI objects.
    The ROI data is assumed to be pre-filtered, so no internal dose thresholding is done.

    Args:
        reference_roi: ROI_Data object for the reference dose (e.g., from MCC).
        evaluation_roi: ROI_Data object for the evaluation dose (e.g., from DICOM).
        dose_percent_threshold: Dose difference criterion (%).
        distance_mm_threshold: DTA criterion (mm).
        global_normalisation: Whether to use global normalization.
        save_csv: Whether to save analysis maps to CSV.
        csv_dir: Directory to save CSV files.

    Returns:
        A tuple containing gamma maps, stats, and other analysis results.
    """
    try:
        # --- Step 1: Extract data from ROI objects ---
        # The core assumption is that the reference data comes from the MCC's original points,
        # which are stored in the metadata.
        if 'original_points' not in reference_roi.source_metadata:
            raise ValueError("Reference ROI must contain 'original_points' in source_metadata for gamma analysis.")

        points_ref = reference_roi.source_metadata['original_points']['coords']
        doses_ref = reference_roi.source_metadata['original_points']['values']

        # --- Step 1a: Filter reference points to be within the reference ROI ---
        ref_extent = reference_roi.physical_extent
        x_min, x_max, y_min, y_max = ref_extent[0], ref_extent[1], ref_extent[2], ref_extent[3]

        mask = (points_ref[:, 0] >= x_min) & (points_ref[:, 0] <= x_max) & \
               (points_ref[:, 1] >= y_min) & (points_ref[:, 1] <= y_max)

        points_ref = points_ref[mask]
        doses_ref = doses_ref[mask]

        if points_ref.size == 0:
            raise ValueError("No valid measurement data points found within the defined ROI.")

        norm_dose = np.max(doses_ref) if global_normalisation else 1.0
        if norm_dose == 0:
            raise ValueError("Cannot determine normalization dose (max reference dose is zero).")

        # --- Step 2: Extract evaluation data from the evaluation ROI ---
        eval_grid = evaluation_roi.dose_grid
        eval_x_coords = evaluation_roi.x_coords
        eval_y_coords = evaluation_roi.y_coords
        phys_extent = evaluation_roi.physical_extent

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
        grid_x, grid_y = np.meshgrid(reference_roi.x_coords, reference_roi.y_coords)

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

        # The interpolated MCC data for visualization is now just the reference ROI's dose grid
        mcc_interp_data = reference_roi.dose_grid

        # --- Step 5: Save maps to CSV (Optional) ---
        if save_csv and csv_dir:
            try:
                base_filename = os.path.splitext(os.path.basename(reference_roi.source_metadata.get('filename', 'mcc')))[0]
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
