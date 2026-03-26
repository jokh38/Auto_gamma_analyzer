"""
This module provides functions for generating reports of the gamma analysis results.
"""
import matplotlib.pyplot as plt
import numpy as np
from src.plot_styles import get_gamma_colormap, get_gamma_norm

def _style_report_axis(ax):
    """Force report axes to use black text on a white background."""
    ax.set_facecolor('white')
    ax.title.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='both', colors='black', labelcolor='black')
    for spine in ax.spines.values():
        spine.set_color('black')

def _style_report_colorbar(colorbar):
    """Force report colorbar text to black."""
    if colorbar is None:
        return
    colorbar.ax.yaxis.label.set_color('black')
    colorbar.ax.tick_params(colors='black', labelcolor='black')
    for spine in colorbar.ax.spines.values():
        spine.set_color('black')

def _style_report_legend(legend):
    """Force report legend text to black."""
    if legend is None:
        return
    frame = legend.get_frame()
    if frame is not None:
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
    for text in legend.get_texts():
        text.set_color('black')

def _add_results_table(ax, dta, dd, suppression_level, pass_rate, total_points,
                       passed_points, failed_points, dd_stats, dta_stats):
    """Render report metrics as a compact table."""
    rows = [
        ["Criteria", ""],
        ["DTA (mm)", f"{dta:g}"],
        ["DD (%)", f"{dd:g}"],
        ["Threshold (%)", f"{suppression_level:g}"],
        ["Gamma", ""],
        ["Pass Rate (%)", f"{pass_rate:.2f}"],
        ["Analyzed", f"{total_points:,}"],
        ["Passed", f"{passed_points:,}"],
        ["Failed", f"{failed_points:,}"],
    ]

    if dd_stats:
        rows.extend([
            ["DD Analysis", ""],
            ["DD Mean (%)", f"{dd_stats.get('mean', 0):.2f}"],
            ["DD Max (%)", f"{dd_stats.get('max', 0):.2f}"],
            ["DD Min (%)", f"{dd_stats.get('min', 0):.2f}"],
            ["DD Std (%)", f"{dd_stats.get('std', 0):.2f}"],
        ])

    if dta_stats:
        rows.extend([
            ["DTA Analysis", ""],
            ["DTA Mean (mm)", f"{dta_stats.get('mean', 0):.2f}"],
            ["DTA Max (mm)", f"{dta_stats.get('max', 0):.2f}"],
            ["DTA Min (mm)", f"{dta_stats.get('min', 0):.2f}"],
            ["DTA Std (mm)", f"{dta_stats.get('std', 0):.2f}"],
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        colLoc='left',
        cellLoc='left',
        colWidths=[0.68, 0.32],
        bbox=[0.02, 0.02, 0.96, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.28)

    section_rows = {"Criteria", "Gamma", "DD Analysis", "DTA Analysis"}
    if pass_rate >= 95.0:
        gamma_row_color = '#eaffea'
    elif pass_rate >= 90.0:
        gamma_row_color = '#fffae6'
    else:
        gamma_row_color = '#ffeaea'

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(0.8)
        cell.get_text().set_color('black')
        if row == 0:
            cell.set_facecolor('#d9e8f2')
            cell.get_text().set_weight('bold')
            continue

        metric = rows[row - 1][0]
        if metric in section_rows:
            cell.set_facecolor('#eef4f8')
            cell.get_text().set_weight('bold')
            if col == 1:
                cell.get_text().set_text("")
        elif metric == "Pass Rate (%)":
            cell.set_facecolor(gamma_row_color)
        else:
            cell.set_facecolor('white')

    ax.axis('off')

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
    fig = plt.figure(figsize=(12, 18), facecolor='white')
    gs = fig.add_gridspec(
        5,
        2,
        height_ratios=[0.45, 1.0, 1.0, 1.0, 1.0],
        width_ratios=[1.0, 1.0]
    )

    # Patient Info Header - Three-line format
    institution, patient_id, patient_name = dicom_handler.get_patient_info()
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')

    # Line 1: Main title
    title_ax.text(0.5, 0.85, '2D Gamma Report',
                 ha='center', va='top', fontsize=24, weight='bold', color='black')

    # Line 2: Patient info
    title_ax.text(0.5, 0.50, f'Patient: {patient_id} / {patient_name}',
                 ha='center', va='center', fontsize=18, weight='bold', color='black')

    # Line 3: Institution
    title_ax.text(0.5, 0.15, f'Institution: {institution}',
                 ha='center', va='bottom', fontsize=16, color='black')

    # 1. 2D Dose Plots
    # DICOM Dose
    ax_dicom = fig.add_subplot(gs[1, 0], facecolor='white')
    dicom_data = dicom_handler.get_pixel_data()
    dicom_extent = dicom_handler.get_physical_extent()
    if dicom_data is not None and dicom_extent is not None:
        im_dicom = ax_dicom.imshow(dicom_data, cmap='jet', extent=dicom_extent, aspect='equal', origin='upper')
        cbar_dicom = fig.colorbar(im_dicom, ax=ax_dicom, label='Dose (Gy)', orientation='vertical', pad=0.02)
        _style_report_colorbar(cbar_dicom)
    ax_dicom.set_title('DICOM RT Dose', fontsize=14, weight='bold')
    ax_dicom.set_xlabel('Position (mm)', fontsize=12)
    ax_dicom.set_ylabel('Position (mm)', fontsize=12)
    ax_dicom.tick_params(labelsize=10)
    _style_report_axis(ax_dicom)

    # MCC Dose
    ax_mcc = fig.add_subplot(gs[1, 1], facecolor='white')
    # mcc_interp_data is on the (cropped) DICOM grid
    if mcc_interp_data is not None and dicom_extent is not None:
        im_mcc = ax_mcc.imshow(mcc_interp_data, cmap='jet', extent=dicom_extent, aspect='equal', origin='upper')
        cbar_mcc = fig.colorbar(im_mcc, ax=ax_mcc, label='Dose (Gy)', orientation='vertical', pad=0.02)
        _style_report_colorbar(cbar_mcc)

    ax_mcc.set_title('MCC Dose (Interpolated)', fontsize=14, weight='bold')
    ax_mcc.set_xlabel('Position (mm)', fontsize=12)
    ax_mcc.set_ylabel('Position (mm)', fontsize=12)
    ax_mcc.tick_params(labelsize=10)
    _style_report_axis(ax_mcc)


    # 2. Profile Plots
    # Horizontal Profile
    ax_hor_profile = fig.add_subplot(gs[2, 0], facecolor='white')
    if hor_profile_data:
        ax_hor_profile.plot(hor_profile_data['phys_coords'], hor_profile_data['dicom_values'], 'b-', linewidth=2, label='RT dose')
        if 'mcc_values' in hor_profile_data and 'mcc_phys_coords' in hor_profile_data:
            ax_hor_profile.plot(hor_profile_data['mcc_phys_coords'], hor_profile_data['mcc_values'], 'ro', markersize=6, alpha=0.7)

        ax_hor_profile.set_xlabel('Position (mm)', fontsize=12)
        ax_hor_profile.set_ylabel('Dose (Gy)', fontsize=12)
        ax_hor_profile.set_title('Left-Right Profile (Horizontal)', fontsize=14, weight='bold')
        ax_hor_profile.tick_params(labelsize=10)
        legend = ax_hor_profile.legend(fontsize=10)
        _style_report_legend(legend)
        ax_hor_profile.grid(False)
    _style_report_axis(ax_hor_profile)

    # Vertical Profile
    ax_ver_profile = fig.add_subplot(gs[2, 1], facecolor='white')
    if ver_profile_data:
        ax_ver_profile.plot(ver_profile_data['phys_coords'], ver_profile_data['dicom_values'], 'b-', linewidth=2, label='RT dose')
        if 'mcc_values' in ver_profile_data and 'mcc_phys_coords' in ver_profile_data:
            ax_ver_profile.plot(ver_profile_data['mcc_phys_coords'], ver_profile_data['mcc_values'], 'ro', markersize=6, alpha=0.7)

        ax_ver_profile.set_xlabel('Position (mm)', fontsize=12)
        ax_ver_profile.set_ylabel('Dose (Gy)', fontsize=12)
        ax_ver_profile.set_title('In-Out Profile (Vertical)', fontsize=14, weight='bold')
        ax_ver_profile.tick_params(labelsize=10)
        legend = ax_ver_profile.legend(fontsize=10)
        _style_report_legend(legend)
        ax_ver_profile.grid(False)
    _style_report_axis(ax_ver_profile)

    # 3. Gamma Analysis
    # Gamma Map - Use interpolated version if available for gap-free visualization
    ax_gamma = fig.add_subplot(gs[3, 0], facecolor='white')
    gamma_cmap = get_gamma_colormap()
    gamma_norm = None
    gamma_source = gamma_map_interp if gamma_map_interp is not None else gamma_map
    if gamma_source is not None:
        finite_values = np.asarray(gamma_source, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        gamma_norm = get_gamma_norm(np.nanmax(finite_values) if finite_values.size else None)

    if gamma_map_interp is not None and dicom_extent is not None:
        # Use interpolated gamma map on DICOM grid (no gaps)
        im_gamma = ax_gamma.imshow(
            gamma_map_interp,
            cmap=gamma_cmap,
            norm=gamma_norm,
            extent=dicom_extent,
            aspect='equal',
            origin='upper',
            interpolation='bilinear'
        )
        cbar_gamma = fig.colorbar(im_gamma, ax=ax_gamma, label='Gamma Index', orientation='vertical', pad=0.02, extend='max')
        _style_report_colorbar(cbar_gamma)
        
        # Add contour at gamma = 1.0
        X = np.linspace(dicom_extent[0], dicom_extent[1], gamma_map_interp.shape[1])
        Y = np.linspace(dicom_extent[3], dicom_extent[2], gamma_map_interp.shape[0])  # Match origin='upper'
        ax_gamma.contour(X, Y, gamma_map_interp, levels=[1.0], colors='black', linewidths=1.5, linestyles='solid', alpha=0.8)
    elif gamma_map is not None:
        # Fallback to sparse gamma map on MCC grid
        mcc_extent = mcc_handler.get_physical_extent()
        if mcc_extent is not None:
            im_gamma = ax_gamma.imshow(
                gamma_map,
                cmap=gamma_cmap,
                norm=gamma_norm,
                extent=mcc_extent,
                aspect='equal',
                origin='upper'
            )
            cbar_gamma = fig.colorbar(im_gamma, ax=ax_gamma, label='Gamma Index', orientation='vertical', pad=0.02, extend='max')
            _style_report_colorbar(cbar_gamma)

    ax_gamma.set_title(f'Gamma Analysis', fontsize=14, weight='bold')
    ax_gamma.set_xlabel('Position (mm)', fontsize=12)
    ax_gamma.set_ylabel('Position (mm)', fontsize=12)
    ax_gamma.tick_params(labelsize=10)
    _style_report_axis(ax_gamma)

    # Results Panel - Table layout
    ax_results = fig.add_subplot(gs[3, 1], facecolor='white')
    ax_results.axis('off')

    pass_rate = gamma_stats.get('pass_rate', 0)
    total_points = gamma_stats.get('total_points', 0)
    passed_points = int(total_points * pass_rate / 100)
    failed_points = total_points - passed_points
    _add_results_table(
        ax_results,
        dta=dta,
        dd=dd,
        suppression_level=suppression_level,
        pass_rate=pass_rate,
        total_points=total_points,
        passed_points=passed_points,
        failed_points=failed_points,
        dd_stats=dd_stats,
        dta_stats=dta_stats,
    )

    # 4. DD and DTA Analysis - Use interpolated versions if available for gap-free visualization
    if dd_map_interp is not None or dd_map is not None:
        # DD Map (Dose Difference)
        ax_dd = fig.add_subplot(gs[4, 0], facecolor='white')
        if dd_map_interp is not None and dicom_extent is not None:
            # Use interpolated DD map on DICOM grid (no gaps)
            im_dd = ax_dd.imshow(dd_map_interp, cmap='viridis', extent=dicom_extent, aspect='equal', origin='upper', interpolation='bilinear')
            cbar_dd = fig.colorbar(im_dd, ax=ax_dd, label='DD (%)', orientation='vertical', pad=0.02, extend='both')
            _style_report_colorbar(cbar_dd)
        elif dd_map is not None:
            # Fallback to sparse DD map on MCC grid
            mcc_extent = mcc_handler.get_physical_extent()
            if mcc_extent is not None:
                im_dd = ax_dd.imshow(dd_map, cmap='viridis', extent=mcc_extent, aspect='equal', origin='upper')
                cbar_dd = fig.colorbar(im_dd, ax=ax_dd, label='DD (%)', orientation='vertical', pad=0.02, extend='both')
                _style_report_colorbar(cbar_dd)

        ax_dd.set_title(f'Dose Difference (DD) Map', fontsize=14, weight='bold')
        ax_dd.set_xlabel('Position (mm)', fontsize=12)
        ax_dd.set_ylabel('Position (mm)', fontsize=12)
        ax_dd.tick_params(labelsize=10)
        _style_report_axis(ax_dd)

    if dta_map_interp is not None or dta_map is not None:
        # DTA Map (Distance to Agreement)
        ax_dta = fig.add_subplot(gs[4, 1], facecolor='white')
        if dta_map_interp is not None and dicom_extent is not None:
            # Use interpolated DTA map on DICOM grid (no gaps)
            im_dta = ax_dta.imshow(dta_map_interp, cmap='plasma', extent=dicom_extent, aspect='equal', origin='upper', interpolation='bilinear')
            cbar_dta = fig.colorbar(im_dta, ax=ax_dta, label='DTA (mm)', orientation='vertical', pad=0.02, extend='max')
            _style_report_colorbar(cbar_dta)
        elif dta_map is not None:
            # Fallback to sparse DTA map on MCC grid
            mcc_extent = mcc_handler.get_physical_extent()
            if mcc_extent is not None:
                im_dta = ax_dta.imshow(dta_map, cmap='plasma', extent=mcc_extent, aspect='equal', origin='upper')
                cbar_dta = fig.colorbar(im_dta, ax=ax_dta, label='DTA (mm)', orientation='vertical', pad=0.02, extend='max')
                _style_report_colorbar(cbar_dta)

        ax_dta.set_title(f'Distance to Agreement (DTA) Map', fontsize=14, weight='bold')
        ax_dta.set_xlabel('Position (mm)', fontsize=12)
        ax_dta.set_ylabel('Position (mm)', fontsize=12)
        ax_dta.tick_params(labelsize=10)
        _style_report_axis(ax_dta)

    plt.tight_layout(rect=(0, 0, 1, 0.97), pad=1.8, w_pad=1.6, h_pad=1.8)
    plt.savefig(output_path, format='jpeg', dpi=300, facecolor='white', edgecolor='white')
    plt.close(fig)
