"""
This module provides functions for generating reports of the gamma analysis results for the GUI.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional

from .standard_data_model import StandardDoseData


def generate_report(
    output_path: str,
    dicom_data: StandardDoseData,
    mcc_data: StandardDoseData,
    gamma_map: np.ndarray,
    gamma_stats: dict,
    dta: float,
    dd: float,
    suppression_level: float,
    ver_profile_data: dict,
    hor_profile_data: dict,
    mcc_interp_data: Optional[np.ndarray] = None,
    dd_stats: Optional[dict] = None,
    dta_stats: Optional[dict] = None,
    dose_bounds: Optional[dict] = None
):
    """
    Generates a comprehensive PDF/JPEG report of the gamma analysis results.
    """
    fig = plt.figure(figsize=(12, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    # Patient Info Header
    patient_name = dicom_data.metadata.get('patient_name', 'N/A')
    patient_id = dicom_data.metadata.get('patient_id', 'N/A')
    institution = dicom_data.metadata.get('institution', 'N/A')
    fig.suptitle(f'Institution: {institution} | Patient: {patient_name} ({patient_id})', fontsize=16)

    # 1. 2D Dose Plots (Row 1)
    # DICOM Dose
    ax_dicom = fig.add_subplot(gs[0, 0])
    im_dicom = ax_dicom.imshow(dicom_data.data_grid, cmap='jet', extent=dicom_data.physical_extent, aspect='equal', origin='lower')
    if dose_bounds:
        ax_dicom.set_xlim(dose_bounds['min_x'], dose_bounds['max_x'])
        ax_dicom.set_ylim(dose_bounds['min_y'], dose_bounds['max_y'])
    fig.colorbar(im_dicom, ax=ax_dicom, label='Dose (Gy)')
    ax_dicom.set_title(f"DICOM: {os.path.basename(dicom_data.metadata.get('filename', ''))}")
    ax_dicom.set_xlabel('Position (mm)')
    ax_dicom.set_ylabel('Position (mm)')

    # MCC Dose
    ax_mcc = fig.add_subplot(gs[0, 1])
    if mcc_interp_data is not None:
        # The interpolated data from analysis is on the DICOM grid, so use its extent
        im_mcc = ax_mcc.imshow(mcc_interp_data, cmap='jet', extent=dicom_data.physical_extent, aspect='equal', origin='lower')
        if dose_bounds:
            ax_mcc.set_xlim(dose_bounds['min_x'], dose_bounds['max_x'])
            ax_mcc.set_ylim(dose_bounds['min_y'], dose_bounds['max_y'])
        fig.colorbar(im_mcc, ax=ax_mcc, label='Dose')
    ax_mcc.set_title(f"MCC: {os.path.basename(mcc_data.metadata.get('filename', ''))} (Interpolated)")
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
            ax_hor_profile.plot(hor_profile_data['mcc_phys_coords'], hor_profile_data['mcc_values'], 'ro', markersize=4, label='MCC points')
        if dose_bounds:
            ax_hor_profile.set_xlim(dose_bounds['min_x'], dose_bounds['max_x'])
        ax_hor_profile.set_title('Horizontal Profile')
        ax_hor_profile.legend()
        ax_hor_profile.grid(True)

    # Vertical Profile
    ax_ver_profile = fig.add_subplot(gs[1, 1])
    if ver_profile_data:
        ax_ver_profile.plot(ver_profile_data['phys_coords'], ver_profile_data['dicom_values'], 'b-', linewidth=2, label='RT dose (x=0)')
        if 'mcc_interp' in ver_profile_data and ver_profile_data['mcc_interp'] is not None:
            ax_ver_profile.plot(ver_profile_data['phys_coords'], ver_profile_data['mcc_interp'], 'r--', linewidth=2, label='mcc dose (x=0)')
        if 'mcc_values' in ver_profile_data and 'mcc_phys_coords' in ver_profile_data:
            ax_ver_profile.plot(ver_profile_data['mcc_phys_coords'], ver_profile_data['mcc_values'], 'ro', markersize=4, label='MCC points')
        if dose_bounds:
            ax_ver_profile.set_xlim(dose_bounds['min_y'], dose_bounds['max_y'])
        ax_ver_profile.set_title('Vertical Profile')
        ax_ver_profile.legend()
        ax_ver_profile.grid(True)

    # 3. Gamma Analysis (Row 3)
    # Gamma Map
    ax_gamma = fig.add_subplot(gs[2, 0])
    if gamma_map is not None:
        im_gamma = ax_gamma.imshow(gamma_map, cmap='coolwarm', extent=mcc_data.physical_extent, vmin=0, vmax=2, aspect='equal', origin='lower')
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
    stats_text = (
        f"Gamma Analysis Results\n\n"
        f"Acceptance Criteria:\n"
        f"  DTA: {dta} mm, DD: {dd} %, Threshold: {suppression_level} %\n\n"
        f"Results:\n"
        f"  Passing Rate: {pass_rate:.2f} %\n"
        f"  Analyzed Points: {total_points}\n"
        f"  Passed Points: {passed_points}\n"
        f"  Failed Points: {failed_points}\n"
    )
    if dd_stats:
        stats_text += f"\nDD Stats: Mean={dd_stats.get('mean', 0):.2f}, Max={dd_stats.get('max', 0):.2f}\n"
    if dta_stats:
        stats_text += f"DTA Stats: Mean={dta_stats.get('mean', 0):.2f}, Max={dta_stats.get('max', 0):.2f}\n"
    ax_gamma_text.text(0.05, 0.95, stats_text, transform=ax_gamma_text.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)