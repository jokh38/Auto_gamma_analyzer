"""
This module provides functions for generating reports of the gamma analysis results.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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
    dd_map=None,
    dta_map=None,
    dd_stats=None,
    dta_stats=None,
    additional_profiles=None,
    gamma_map_interp=None,
    dd_map_interp=None,
    dta_map_interp=None
):
    """
    Generates a comprehensive PDF report of the gamma analysis results.

    The report includes:
    - Patient and file information.
    - 2D dose distribution plots for both DICOM and MCC data.
    - Dose profile comparisons (horizontal and vertical).
    - A 2D gamma map, with statistics.
    - Dose Difference (DD) and Distance-to-Agreement (DTA) maps.

    Args:
        output_path (str): The path to save the generated report file.
        dicom_handler (DicomFileHandler): The handler for the DICOM data.
        mcc_handler (MCCFileHandler): The handler for the MCC data.
        gamma_map (np.ndarray): The calculated gamma map (sparse, on MCC grid).
        gamma_stats (dict): A dictionary of statistics for the gamma analysis.
        dta (float): The Distance-to-Agreement criterion used (in mm).
        dd (float): The Dose Difference criterion used (in %).
        suppression_level (float): The dose threshold for the analysis (in %).
        ver_profile_data (dict): Data for the vertical dose profile.
        hor_profile_data (dict): Data for the horizontal dose profile.
        mcc_interp_data (np.ndarray, optional): Interpolated MCC data. Defaults to None.
        dd_map (np.ndarray, optional): The dose difference map (sparse). Defaults to None.
        dta_map (np.ndarray, optional): The distance-to-agreement map (sparse). Defaults to None.
        dd_stats (dict, optional): Statistics for the dose difference analysis. Defaults to None.
        dta_stats (dict, optional): Statistics for the distance-to-agreement analysis. Defaults to None.
        additional_profiles (dict, optional): Data for additional dose profiles. Defaults to None.
        gamma_map_interp (np.ndarray, optional): Interpolated gamma map (on DICOM grid). Defaults to None.
        dd_map_interp (np.ndarray, optional): Interpolated DD map (on DICOM grid). Defaults to None.
        dta_map_interp (np.ndarray, optional): Interpolated DTA map (on DICOM grid). Defaults to None.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.5, 1, 1], width_ratios=[1, 1, 1, 1])

    # Patient Info Header - Three-line format
    institution, patient_id, patient_name = dicom_handler.get_patient_info()
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')

    # Line 1: Main title
    title_ax.text(0.5, 0.85, '2D Gamma Report',
                 ha='center', va='top', fontsize=22, weight='bold')

    # Line 2: Patient info
    title_ax.text(0.5, 0.50, f'Patient: {patient_id} / {patient_name}',
                 ha='center', va='center', fontsize=18, weight='bold')

    # Line 3: Institution
    title_ax.text(0.5, 0.15, f'Institution: {institution}',
                 ha='center', va='bottom', fontsize=16)

    # 1. 2D Dose Plots
    # DICOM Dose
    ax_dicom = fig.add_subplot(gs[1, 0])
    dicom_data = dicom_handler.get_pixel_data()
    dicom_extent = dicom_handler.get_physical_extent()
    if dicom_data is not None and dicom_extent is not None:
        im_dicom = ax_dicom.imshow(dicom_data, cmap='jet', extent=dicom_extent, aspect='equal', origin='upper')
        cbar_dicom = fig.colorbar(im_dicom, ax=ax_dicom, label='Dose (Gy)', orientation='vertical', pad=0.02)
    ax_dicom.set_title('DICOM RT Dose', fontsize=12, weight='bold')
    ax_dicom.set_xlabel('Position (mm)', fontsize=10)
    ax_dicom.set_ylabel('Position (mm)', fontsize=10)

    # MCC Dose
    ax_mcc = fig.add_subplot(gs[1, 1])
    # mcc_interp_data is on the (cropped) DICOM grid
    if mcc_interp_data is not None and dicom_extent is not None:
        im_mcc = ax_mcc.imshow(mcc_interp_data, cmap='jet', extent=dicom_extent, aspect='equal', origin='upper')
        cbar_mcc = fig.colorbar(im_mcc, ax=ax_mcc, label='Dose (Gy)', orientation='vertical', pad=0.02)

    ax_mcc.set_title('MCC Dose (Interpolated)', fontsize=12, weight='bold')
    ax_mcc.set_xlabel('Position (mm)', fontsize=10)
    ax_mcc.set_ylabel('Position (mm)', fontsize=10)


    # 2. Profile Plots
    # Horizontal Profile
    ax_hor_profile = fig.add_subplot(gs[2, 0])
    if hor_profile_data:
        ax_hor_profile.plot(hor_profile_data['phys_coords'], hor_profile_data['dicom_values'], 'b-', linewidth=2, label='RT dose')
        if 'mcc_interp' in hor_profile_data:
            ax_hor_profile.plot(hor_profile_data['phys_coords'], hor_profile_data['mcc_interp'], 'r--', linewidth=2, label='mcc dose')
        if 'mcc_values' in hor_profile_data and 'mcc_phys_coords' in hor_profile_data:
            ax_hor_profile.plot(hor_profile_data['mcc_phys_coords'], hor_profile_data['mcc_values'], 'ro', markersize=8)

        ax_hor_profile.set_xlabel('Position (mm)', fontsize=10)
        ax_hor_profile.set_ylabel('Dose (Gy)', fontsize=10)
        ax_hor_profile.set_title('Left-Right Profile (Horizontal)', fontsize=12, weight='bold')
        ax_hor_profile.legend()
        ax_hor_profile.grid(True)

    # Vertical Profile
    ax_ver_profile = fig.add_subplot(gs[2, 1])
    if ver_profile_data:
        ax_ver_profile.plot(ver_profile_data['phys_coords'], ver_profile_data['dicom_values'], 'b-', linewidth=2, label='RT dose')
        if 'mcc_interp' in ver_profile_data:
            ax_ver_profile.plot(ver_profile_data['phys_coords'], ver_profile_data['mcc_interp'], 'r--', linewidth=2, label='mcc dose')
        if 'mcc_values' in ver_profile_data and 'mcc_phys_coords' in ver_profile_data:
            ax_ver_profile.plot(ver_profile_data['mcc_phys_coords'], ver_profile_data['mcc_values'], 'ro', markersize=8)

        ax_ver_profile.set_xlabel('Position (mm)', fontsize=10)
        ax_ver_profile.set_ylabel('Dose (Gy)', fontsize=10)
        ax_ver_profile.set_title('In-Out Profile (Vertical)', fontsize=12, weight='bold')
        ax_ver_profile.legend()
        ax_ver_profile.grid(True)

    # 3. Gamma Analysis
    # Gamma Map - Use interpolated version if available for gap-free visualization
    ax_gamma = fig.add_subplot(gs[1, 2])
    if gamma_map_interp is not None and dicom_extent is not None:
        # Use interpolated gamma map on DICOM grid (no gaps)
        im_gamma = ax_gamma.imshow(gamma_map_interp, cmap='coolwarm', extent=dicom_extent, vmin=0, vmax=2, aspect='equal', origin='upper', interpolation='bilinear')
        cbar_gamma = fig.colorbar(im_gamma, ax=ax_gamma, label='Gamma Index', orientation='vertical', pad=0.02)
    elif gamma_map is not None:
        # Fallback to sparse gamma map on MCC grid
        mcc_extent = mcc_handler.get_physical_extent()
        if mcc_extent is not None:
            im_gamma = ax_gamma.imshow(gamma_map, cmap='coolwarm', extent=mcc_extent, vmin=0, vmax=2, aspect='equal', origin='upper')
            cbar_gamma = fig.colorbar(im_gamma, ax=ax_gamma, label='Gamma Index', orientation='vertical', pad=0.02)

    ax_gamma.set_title(f'Gamma Analysis', fontsize=12, weight='bold')
    ax_gamma.set_xlabel('Position (mm)', fontsize=10)
    ax_gamma.set_ylabel('Position (mm)', fontsize=10)

    # Results Panel - Two-column layout
    ax_results = fig.add_subplot(gs[1, 3])
    ax_results.axis('off')

    pass_rate = gamma_stats.get('pass_rate', 0)
    total_points = gamma_stats.get('total_points', 0)
    passed_points = int(total_points * pass_rate / 100)
    failed_points = total_points - passed_points

    # Left Panel: Gamma criteria
    gamma_text = (
        f"═ Criteria ═\n"
        f" •DTA: {dta} mm\n"
        f" •DD: {dd}%\n"
        f" •Threshold: {suppression_level}%"
    )

    ax_results.text(0.02, 0.98, gamma_text, transform=ax_results.transAxes, fontsize=10,
                   verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.6', fc='aliceblue', alpha=0.85,
                            edgecolor='steelblue', linewidth=1.5))
    
    # Left Panel: Gamma Analysis
    gamma_text = (
        f"═ γ Analysis ═\n\n"
        f" ►Result:{pass_rate:.2f}%\n\n"
        f" •Analyzed: {total_points:,}\n"
        f"  •Passed: {passed_points:,}\n"
        f"  •Failed: {failed_points:,}"
    )

    ax_results.text(0.02, 0.4, gamma_text, transform=ax_results.transAxes, fontsize=13,
                   verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.6', fc='lightyellow', alpha=0.85,
                             edgecolor='orange', linewidth=1.5))

    # Right Panel: DD & DTA Analysis
    dd_dta_text = ""
    if dd_stats:
        dd_dta_text += (
            f"═ DD Analysis(%) ═\n"
            f" •Mean: {dd_stats.get('mean', 0):>6.2f}\n"
            f" •Max : {dd_stats.get('max', 0):>6.2f}\n"
            f" •Min : {dd_stats.get('min', 0):>6.2f}\n"
            f" •Std : {dd_stats.get('std', 0):>6.2f}\n"
        )

    if dta_stats:
        if dd_dta_text:
            dd_dta_text += "\n"
        dd_dta_text += (
            f"═ DTA Analysis(mm) ═\n"
            f" •Mean: {dta_stats.get('mean', 0):>6.2f}\n"
            f" •Max : {dta_stats.get('max', 0):>6.2f}\n"
            f" •Min : {dta_stats.get('min', 0):>6.2f}\n"
            f" •Std : {dta_stats.get('std', 0):>6.2f}"
        )

    if dd_dta_text:
        ax_results.text(0.52, 0.98, dd_dta_text, transform=ax_results.transAxes, fontsize=10,
                       verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round,pad=0.6', fc='aliceblue', alpha=0.85,
                                 edgecolor='steelblue', linewidth=1.5))

    # 4. DD and DTA Analysis - Use interpolated versions if available for gap-free visualization
    if dd_map_interp is not None or dd_map is not None:
        # DD Map (Dose Difference)
        ax_dd = fig.add_subplot(gs[2, 2])
        if dd_map_interp is not None and dicom_extent is not None:
            # Use interpolated DD map on DICOM grid (no gaps)
            im_dd = ax_dd.imshow(dd_map_interp, cmap='viridis', extent=dicom_extent, aspect='equal', origin='upper', interpolation='bilinear')
            cbar_dd = fig.colorbar(im_dd, ax=ax_dd, label='DD (%)', orientation='vertical', pad=0.02)
        elif dd_map is not None:
            # Fallback to sparse DD map on MCC grid
            mcc_extent = mcc_handler.get_physical_extent()
            if mcc_extent is not None:
                im_dd = ax_dd.imshow(dd_map, cmap='viridis', extent=mcc_extent, aspect='equal', origin='upper')
                cbar_dd = fig.colorbar(im_dd, ax=ax_dd, label='DD (%)', orientation='vertical', pad=0.02)

        ax_dd.set_title(f'Dose Difference (DD) Map', fontsize=12, weight='bold')
        ax_dd.set_xlabel('Position (mm)', fontsize=10)
        ax_dd.set_ylabel('Position (mm)', fontsize=10)

    if dta_map_interp is not None or dta_map is not None:
        # DTA Map (Distance to Agreement)
        ax_dta = fig.add_subplot(gs[2, 3])
        if dta_map_interp is not None and dicom_extent is not None:
            # Use interpolated DTA map on DICOM grid (no gaps)
            im_dta = ax_dta.imshow(dta_map_interp, cmap='plasma', extent=dicom_extent, aspect='equal', origin='upper', interpolation='bilinear')
            cbar_dta = fig.colorbar(im_dta, ax=ax_dta, label='DTA (mm)', orientation='vertical', pad=0.02)
        elif dta_map is not None:
            # Fallback to sparse DTA map on MCC grid
            mcc_extent = mcc_handler.get_physical_extent()
            if mcc_extent is not None:
                im_dta = ax_dta.imshow(dta_map, cmap='plasma', extent=mcc_extent, aspect='equal', origin='upper')
                cbar_dta = fig.colorbar(im_dta, ax=ax_dta, label='DTA (mm)', orientation='vertical', pad=0.02)

        ax_dta.set_title(f'Distance to Agreement (DTA) Map', fontsize=12, weight='bold')
        ax_dta.set_xlabel('Position (mm)', fontsize=10)
        ax_dta.set_ylabel('Position (mm)', fontsize=10)

    plt.tight_layout(rect=(0, 0, 1, 0.96), pad=2.0, w_pad=2.5, h_pad=2.5)
    plt.savefig(output_path, format='jpeg', dpi=150)
    plt.close(fig)
