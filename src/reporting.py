"""
This module provides functions for generating reports of the gamma analysis results for the GUI.
"""
import matplotlib.pyplot as plt
import numpy as np


def generate_report(
    output_path,
    dicom_handler,
    mcc_handler,
    gamma_map,
    gamma_stats,
    dta,
    dd,
    suppression_level,
    ver_profile_data,
    hor_profile_data,
    mcc_interp_data=None,
    dd_stats=None,
    dta_stats=None,
    additional_profiles=None,
    dose_bounds=None
):
    """
    Generates a comprehensive PDF/JPEG report of the gamma analysis results.

    The report includes:
    - Patient and file information.
    - 2D dose distribution plots for both DICOM and MCC data.
    - Dose profile comparisons (horizontal and vertical).
    - A 2D gamma map, with statistics.

    Args:
        output_path (str): The path to save the generated report file (e.g., 'report.jpg' or 'report.pdf').
        dicom_handler (DicomFileHandler): The handler for the DICOM data.
        mcc_handler (MCCFileHandler): The handler for the MCC data.
        gamma_map (np.ndarray): The calculated gamma map.
        gamma_stats (dict): A dictionary of statistics for the gamma analysis.
        dta (float): The Distance-to-Agreement criterion used (in mm).
        dd (float): The Dose Difference criterion used (in %).
        suppression_level (float): The dose threshold for the analysis (in %).
        ver_profile_data (dict): Data for the vertical dose profile.
        hor_profile_data (dict): Data for the horizontal dose profile.
        mcc_interp_data (np.ndarray, optional): Interpolated MCC data. Defaults to None.
        dd_stats (dict, optional): Statistics for the dose difference analysis. Defaults to None.
        dta_stats (dict, optional): Statistics for the distance-to-agreement analysis. Defaults to None.
        additional_profiles (dict, optional): Data for additional dose profiles. Defaults to None.
        dose_bounds (dict, optional): Cropping bounds for the dose data. Defaults to None.
    """
    # Create a 3x2 grid for the plots. Figure size is adjusted for the reduced number of rows.
    fig = plt.figure(figsize=(12, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    # Patient Info Header
    institution, patient_id, patient_name = dicom_handler.get_patient_info()
    fig.suptitle(f'Institution: {institution} | Patient: {patient_name} ({patient_id})', fontsize=16)

    # 1. 2D Dose Plots (Row 1)
    # DICOM Dose
    ax_dicom = fig.add_subplot(gs[0, 0])
    dicom_data = dicom_handler.get_pixel_data()
    dicom_extent = dicom_handler.get_physical_extent()
    if dicom_data is not None and dicom_extent is not None:
        im_dicom = ax_dicom.imshow(dicom_data, cmap='jet', extent=dicom_extent, aspect='equal', origin='lower')
        if dose_bounds:
            ax_dicom.set_xlim(dose_bounds['min_x'], dose_bounds['max_x'])
            ax_dicom.set_ylim(dose_bounds['min_y'], dose_bounds['max_y'])
        fig.colorbar(im_dicom, ax=ax_dicom, label='Dose (Gy)')
    ax_dicom.set_title('DICOM RT Dose')
    ax_dicom.set_xlabel('Position (mm)')
    ax_dicom.set_ylabel('Position (mm)')

    # MCC Dose
    ax_mcc = fig.add_subplot(gs[0, 1])
    # mcc_interp_data is on the (cropped) DICOM grid
    if mcc_interp_data is not None and dicom_extent is not None:
        im_mcc = ax_mcc.imshow(mcc_interp_data, cmap='jet', extent=dicom_extent, aspect='equal', origin='lower')
        if dose_bounds:
            ax_mcc.set_xlim(dose_bounds['min_x'], dose_bounds['max_x'])
            ax_mcc.set_ylim(dose_bounds['min_y'], dose_bounds['max_y'])
        fig.colorbar(im_mcc, ax=ax_mcc, label='Dose')

    ax_mcc.set_title('MCC Dose (Interpolated)')
    ax_mcc.set_xlabel('Position (mm)')
    ax_mcc.set_ylabel('Position (mm)')

    # 2. Profile Plots (Row 2)
    # Horizontal Profile
    ax_hor_profile = fig.add_subplot(gs[1, 0])
    if hor_profile_data:
        ax_hor_profile.plot(hor_profile_data['phys_coords'], hor_profile_data['dicom_values'], 'b-', linewidth=2, label='RT dose (y=0)')
        if 'mcc_interp' in hor_profile_data and hor_profile_data['mcc_interp'] is not None:
            ax_hor_profile.plot(hor_profile_data['phys_coords'], hor_profile_data['mcc_interp'], 'r--', linewidth=2, label='mcc dose (y=0)')
        if 'mcc_values' in hor_profile_data and 'mcc_phys_coords' in hor_profile_data:
            ax_hor_profile.plot(hor_profile_data['mcc_phys_coords'], hor_profile_data['mcc_values'], 'ro', markersize=8)

        if dose_bounds:
            ax_hor_profile.set_xlim(dose_bounds['min_x'], dose_bounds['max_x'])

        ax_hor_profile.set_xlabel('Position (mm)')
        ax_hor_profile.set_ylabel('Dose (Gy)')
        ax_hor_profile.set_title('Left-Right Profile (Horizontal)')
        ax_hor_profile.legend()
        ax_hor_profile.grid(True)

    # Vertical Profile
    ax_ver_profile = fig.add_subplot(gs[1, 1])
    if ver_profile_data:
        ax_ver_profile.plot(ver_profile_data['phys_coords'], ver_profile_data['dicom_values'], 'b-', linewidth=2, label='RT dose (x=0)')
        if 'mcc_interp' in ver_profile_data and ver_profile_data['mcc_interp'] is not None:
            ax_ver_profile.plot(ver_profile_data['phys_coords'], ver_profile_data['mcc_interp'], 'r--', linewidth=2, label='mcc dose (x=0)')
        if 'mcc_values' in ver_profile_data and 'mcc_phys_coords' in ver_profile_data:
            ax_ver_profile.plot(ver_profile_data['mcc_phys_coords'], ver_profile_data['mcc_values'], 'ro', markersize=8)

        if dose_bounds:
            ax_ver_profile.set_xlim(dose_bounds['min_y'], dose_bounds['max_y'])

        ax_ver_profile.set_xlabel('Position (mm)')
        ax_ver_profile.set_ylabel('Dose (Gy)')
        ax_ver_profile.set_title('In-Out Profile (Vertical)')
        ax_ver_profile.legend()
        ax_ver_profile.grid(True)

    # 3. Gamma Analysis (Row 3)
    # Gamma Map
    ax_gamma = fig.add_subplot(gs[2, 0])
    mcc_extent = mcc_handler.get_physical_extent()
    if gamma_map is not None and mcc_extent is not None:
        # UI의 draw_image와 일관성을 맞추기 위해, 데이터를 상하 반전하고 origin='lower'를 사용합니다.
        # 이렇게 하면 모든 2D 맵의 Y축이 위쪽을 향하게 됩니다.
        gamma_map_display = np.flipud(gamma_map)
        im_gamma = ax_gamma.imshow(gamma_map_display, cmap='coolwarm', extent=mcc_extent, vmin=0, vmax=2, aspect='equal', origin='lower')
        if dose_bounds:
            ax_gamma.set_xlim(dose_bounds['min_x'], dose_bounds['max_x'])
            ax_gamma.set_ylim(dose_bounds['min_y'], dose_bounds['max_y'])
        fig.colorbar(im_gamma, ax=ax_gamma, label='Gamma Index')
    ax_gamma.set_title(f'Gamma Analysis (Pass: {gamma_stats.get("pass_rate", 0):.1f}%)')
    ax_gamma.set_xlabel('Position (mm)')
    ax_gamma.set_ylabel('Position (mm)')

    # Gamma Stats Text
    ax_gamma_text = fig.add_subplot(gs[2, 1])
    ax_gamma_text.axis('off')

    pass_rate = gamma_stats.get('pass_rate', 0)
    total_points = gamma_stats.get('total_points', 0)
    passed_points = int(total_points * pass_rate / 100)
    failed_points = total_points - passed_points

    # Create comprehensive statistics text including gamma, DD, and DTA
    stats_text = (
        f"Gamma Analysis Results\n\n"
        f"Acceptance Criteria:\n"
        f"  DTA: {dta} mm\n"
        f"  Dose Difference: {dd} %\n"
        f"  Threshold: {suppression_level} %\n\n"
        f"Results:\n"
        f"  Passing Rate: {pass_rate:.2f} %\n"
        f"  Analyzed Pixels: {total_points}\n"
        f"  Passed Pixels: {passed_points}\n"
        f"  Failed Pixels: {failed_points}\n"
    )

    if dd_stats:
        stats_text += (
            f"\nDD Analysis:\n"
            f"  Mean: {dd_stats.get('mean', 0):.2f}\n"
            f"  Max: {dd_stats.get('max', 0):.2f}\n"
            f"  Min: {dd_stats.get('min', 0):.2f}\n"
            f"  Std: {dd_stats.get('std', 0):.2f}\n"
        )

    if dta_stats:
        stats_text += (
            f"\nDTA Analysis:\n"
            f"  Mean: {dta_stats.get('mean', 0):.2f}\n"
            f"  Max: {dta_stats.get('max', 0):.2f}\n"
            f"  Min: {dta_stats.get('min', 0):.2f}\n"
            f"  Std: {dta_stats.get('std', 0):.2f}\n"
        )

    ax_gamma_text.text(0.05, 0.95, stats_text, transform=ax_gamma_text.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.5))

    # Final layout adjustments and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save as JPEG or PDF based on output_path extension
    file_format = output_path.split('.')[-1].lower()
    if file_format not in ['jpeg', 'jpg', 'pdf']:
        file_format = 'jpeg'  # default

    plt.savefig(output_path, format=file_format, dpi=300)
    plt.close(fig)