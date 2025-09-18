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
    additional_profiles=None
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
        gamma_map (np.ndarray): The calculated gamma map.
        gamma_stats (dict): A dictionary of statistics for the gamma analysis.
        dta (float): The Distance-to-Agreement criterion used (in mm).
        dd (float): The Dose Difference criterion used (in %).
        suppression_level (float): The dose threshold for the analysis (in %).
        ver_profile_data (dict): Data for the vertical dose profile.
        hor_profile_data (dict): Data for the horizontal dose profile.
        mcc_interp_data (np.ndarray, optional): Interpolated MCC data. Defaults to None.
        dd_map (np.ndarray, optional): The dose difference map. Defaults to None.
        dta_map (np.ndarray, optional): The distance-to-agreement map. Defaults to None.
        dd_stats (dict, optional): Statistics for the dose difference analysis. Defaults to None.
        dta_stats (dict, optional): Statistics for the distance-to-agreement analysis. Defaults to None.
        additional_profiles (dict, optional): Data for additional dose profiles. Defaults to None.
    """
    fig = plt.figure(figsize=(12, 24))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])

    # Patient Info Header
    institution, patient_id, patient_name = dicom_handler.get_patient_info()
    fig.suptitle(f'Institution: {institution} | Patient: {patient_name} ({patient_id})', fontsize=16)

    # 1. 2D Dose Plots
    # DICOM Dose
    ax_dicom = fig.add_subplot(gs[0, 0])
    dicom_data = dicom_handler.get_pixel_data()
    dicom_extent = dicom_handler.get_physical_extent()
    if dicom_data is not None and dicom_extent is not None:
        im_dicom = ax_dicom.imshow(dicom_data, cmap='jet', extent=dicom_extent, aspect='equal', origin='upper')
        fig.colorbar(im_dicom, ax=ax_dicom, label='Dose (Gy)')
    ax_dicom.set_title('DICOM RT Dose')
    ax_dicom.set_xlabel('Position (mm)')
    ax_dicom.set_ylabel('Position (mm)')

    # MCC Dose
    ax_mcc = fig.add_subplot(gs[0, 1])
    # mcc_interp_data is on the (cropped) DICOM grid
    if mcc_interp_data is not None and dicom_extent is not None:
        im_mcc = ax_mcc.imshow(mcc_interp_data, cmap='jet', extent=dicom_extent, aspect='equal', origin='upper')
        fig.colorbar(im_mcc, ax=ax_mcc, label='Dose')

    ax_mcc.set_title('MCC Dose (Interpolated)')
    ax_mcc.set_xlabel('Position (mm)')
    ax_mcc.set_ylabel('Position (mm)')


    # 2. Profile Plots
    # Horizontal Profile
    ax_hor_profile = fig.add_subplot(gs[1, 0])
    if hor_profile_data:
        ax_hor_profile.plot(hor_profile_data['phys_coords'], hor_profile_data['dicom_values'], 'b-', linewidth=2, label='RT dose (y=0)')
        if 'mcc_interp' in hor_profile_data:
            ax_hor_profile.plot(hor_profile_data['phys_coords'], hor_profile_data['mcc_interp'], 'r--', linewidth=2, label='mcc dose (y=0)')
        if 'mcc_values' in hor_profile_data and 'mcc_phys_coords' in hor_profile_data:
            ax_hor_profile.plot(hor_profile_data['mcc_phys_coords'], hor_profile_data['mcc_values'], 'ro', markersize=8)
            
        # # Add shifted profiles if available
        # if additional_profiles:
        #     if 'hor_plus1' in additional_profiles and additional_profiles['hor_plus1']:
        #         data = additional_profiles['hor_plus1']
        #         ax_hor_profile.plot(data['phys_coords'], data['dicom_values'], 'b:', alpha=0.7, label='RT dose (y=+1cm)')
        #         if 'mcc_interp' in data:
        #             ax_hor_profile.plot(data['phys_coords'], data['mcc_interp'], 'r:', alpha=0.7, label='mcc dose (y=+1cm)')
            
        #     if 'hor_minus1' in additional_profiles and additional_profiles['hor_minus1']:
        #         data = additional_profiles['hor_minus1']
        #         ax_hor_profile.plot(data['phys_coords'], data['dicom_values'], 'b-.', alpha=0.7, label='RT dose (y=-1cm)')
        #         if 'mcc_interp' in data:
        #             ax_hor_profile.plot(data['phys_coords'], data['mcc_interp'], 'r-.', alpha=0.7, label='mcc dose (y=-1cm)')
        
        ax_hor_profile.set_xlabel('Position (mm)')
        ax_hor_profile.set_ylabel('Dose (Gy)')
        ax_hor_profile.set_title('Left-Right Profile (Horizontal)')
        ax_hor_profile.legend()
        ax_hor_profile.grid(True)

    # Vertical Profile
    ax_ver_profile = fig.add_subplot(gs[1, 1])
    if ver_profile_data:
        ax_ver_profile.plot(ver_profile_data['phys_coords'], ver_profile_data['dicom_values'], 'b-', linewidth=2, label='RT dose (x=0)')
        if 'mcc_interp' in ver_profile_data:
            ax_ver_profile.plot(ver_profile_data['phys_coords'], ver_profile_data['mcc_interp'], 'r--', linewidth=2, label='mcc dose (x=0)')
        if 'mcc_values' in ver_profile_data and 'mcc_phys_coords' in ver_profile_data:
            ax_ver_profile.plot(ver_profile_data['mcc_phys_coords'], ver_profile_data['mcc_values'], 'ro', markersize=8)
            
        # # Add shifted profiles if available
        # if additional_profiles:
        #     if 'ver_plus1' in additional_profiles and additional_profiles['ver_plus1']:
        #         data = additional_profiles['ver_plus1']
        #         ax_ver_profile.plot(data['phys_coords'], data['dicom_values'], 'b:', alpha=0.7, label='RT dose (x=+1cm)')
        #         if 'mcc_interp' in data:
        #             ax_ver_profile.plot(data['phys_coords'], data['mcc_interp'], 'r:', alpha=0.7, label='mcc dose (x=+1cm)')
            
        #     if 'ver_minus1' in additional_profiles and additional_profiles['ver_minus1']:
        #         data = additional_profiles['ver_minus1']
        #         ax_ver_profile.plot(data['phys_coords'], data['dicom_values'], 'b-.', alpha=0.7, label='RT dose (x=-1cm)')
        #         if 'mcc_interp' in data:
        #             ax_ver_profile.plot(data['phys_coords'], data['mcc_interp'], 'r-.', alpha=0.7, label='mcc dose (x=-1cm)')
                    
        ax_ver_profile.set_xlabel('Position (mm)')
        ax_ver_profile.set_ylabel('Dose (Gy)')
        ax_ver_profile.set_title('In-Out Profile (Vertical)')
        ax_ver_profile.legend()
        ax_ver_profile.grid(True)

    # 3. Gamma Analysis
    # Gamma Map
    ax_gamma = fig.add_subplot(gs[2, 0])
    mcc_extent = mcc_handler.get_physical_extent()
    if gamma_map is not None and mcc_extent is not None:
        # Flip gamma_map vertically and use origin='lower' to align with other dose distributions
        im_gamma = ax_gamma.imshow(gamma_map, cmap='coolwarm', extent=mcc_extent, vmin=0, vmax=2, aspect='equal', origin='lower')
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
    gamma_text = (
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
    
    dd_text = ""
    if dd_stats:
        dd_text = (
            f"\nDD Analysis:\n"
            f"  Mean: {dd_stats.get('mean', 0):.2f} \n"
            f"  Max: {dd_stats.get('max', 0):.2f} \n"
            f"  Min: {dd_stats.get('min', 0):.2f} \n"
            f"  Std: {dd_stats.get('std', 0):.2f} \n"
        )
    
    dta_text = ""
    if dta_stats:
        dta_text = (
            f"\nDTA Analysis:\n"
            f"  Mean: {dta_stats.get('mean', 0):.2f} \n"
            f"  Max: {dta_stats.get('max', 0):.2f} \n"
            f"  Min: {dta_stats.get('min', 0):.2f} \n"
            f"  Std: {dta_stats.get('std', 0):.2f} \n"
        )
    
    stats_text = gamma_text + dd_text + dta_text
    ax_gamma_text.text(0.05, 0.95, stats_text, transform=ax_gamma_text.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.5))

    # 4. DD and DTA Analysis (4th row)
    if dd_map is not None and dta_map is not None:
        mcc_extent = mcc_handler.get_physical_extent()
        
        # DD Map (Dose Difference)
        ax_dd = fig.add_subplot(gs[3, 0])
        if mcc_extent is not None:
            im_dd = ax_dd.imshow(dd_map, cmap='viridis', extent=mcc_extent, aspect='equal', origin='lower')
            fig.colorbar(im_dd, ax=ax_dd, label='DD')
        ax_dd.set_title(f'Dose Difference (DD) Map')
        ax_dd.set_xlabel('Position (mm)')
        ax_dd.set_ylabel('Position (mm)')
        
        # DTA Map (Distance to Agreement)
        ax_dta = fig.add_subplot(gs[3, 1])
        if mcc_extent is not None:
            im_dta = ax_dta.imshow(dta_map, cmap='plasma', extent=mcc_extent, aspect='equal', origin='lower')
            fig.colorbar(im_dta, ax=ax_dta, label='DTA')
        ax_dta.set_title(f'Distance to Agreement (DTA) Map')
        ax_dta.set_xlabel('Position (mm)')
        ax_dta.set_ylabel('Position (mm)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, format='jpeg')
    plt.close(fig)
