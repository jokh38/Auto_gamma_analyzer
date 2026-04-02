"""
This module provides functions for generating reports of the gamma analysis results.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from src.plot_styles import get_gamma_colormap, get_gamma_norm

# Professional color palette
_HEADER_BG = '#1a3c5e'
_BORDER = '#d0d5dd'
_SECTION_HEADER_BG = '#eef2f7'
_FOOTER_BG = '#f1f5f9'
_MUTED_TEXT = '#64748b'


def _style_report_axis(ax):
    """Force report axes to use black text on a white background."""
    ax.set_facecolor('white')
    ax.title.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='both', colors='black', labelcolor='black')
    for spine in ax.spines.values():
        spine.set_color('black')


def _style_report_colorbar(cbar):
    """Force report colorbar text to black."""
    if cbar is None:
        return
    cbar.ax.yaxis.label.set_color('black')
    cbar.ax.tick_params(colors='black', labelcolor='black', labelsize=10)
    for spine in cbar.ax.spines.values():
        spine.set_color('black')


def _style_report_legend(legend):
    """Force report legend text to black on white."""
    if legend is None:
        return
    frame = legend.get_frame()
    if frame is not None:
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
    for text in legend.get_texts():
        text.set_color('black')


def _draw_header(fig, gs_slot, institution, patient_id, patient_name):
    """Draw a professional dark header bar with patient info (no date)."""
    ax = fig.add_subplot(gs_slot)
    ax.set_facecolor(_HEADER_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Solid background patch to ensure full coverage
    bg = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle=mpatches.BoxStyle("Square", pad=0),
        facecolor=_HEADER_BG, edgecolor='none',
        transform=ax.transAxes, zorder=0)
    ax.add_patch(bg)

    ax.text(0.5, 0.68, '2D Gamma Analysis Report',
            ha='center', va='center', fontsize=24, weight='bold',
            color='#ffffff', zorder=1)
    ax.text(0.5, 0.28,
            f'{patient_id}   \u2502   {patient_name}   \u2502   {institution}',
            ha='center', va='center', fontsize=14, weight='bold',
            color='#ffffff', zorder=1)


def _draw_gamma_summary_table(ax, gamma_stats):
    """Draw a compact gamma summary table."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor('white')
    ax.axis('off')

    total_reference_points = int(gamma_stats.get('total_reference_points', gamma_stats.get('total_points', 0)) or 0)
    evaluated_points = int(gamma_stats.get('evaluated_points', gamma_stats.get('total_points', 0)) or 0)
    passed_points = int(gamma_stats.get('passed_points', 0) or 0)
    failed_points = int(gamma_stats.get('failed_points', max(evaluated_points - passed_points, 0)) or 0)
    def _fmt(count, denominator):
        proportion = 0.0 if denominator <= 0 else (100.0 * count / denominator)
        return f"{count:d}", f"{proportion:.1f}%"

    rows = [
        ["MCC dose points", *_fmt(total_reference_points, total_reference_points or 1)],
        ["Evaluated dose points", *_fmt(evaluated_points, total_reference_points or 1)],
        ["Passed", *_fmt(passed_points, evaluated_points or 1)],
        ["Failed", *_fmt(failed_points, evaluated_points or 1)],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=["Gamma Result", "Count", "Proportion"],
        colLoc='left',
        cellLoc='left',
        colWidths=[0.54, 0.20, 0.26],
        bbox=[0.03, 0.08, 0.94, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(_BORDER)
        cell.set_linewidth(0.6)
        cell.get_text().set_color('black')
        if row == 0:
            cell.set_facecolor(_HEADER_BG)
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            label = rows[row - 1][0]
            if label == "Pass rate":
                if col == 0:
                    cell.set_facecolor(_SECTION_HEADER_BG)
                    cell.get_text().set_weight('bold')
                else:
                    cell.set_facecolor('white')
            elif label == "Passed":
                cell.set_facecolor('#f0fdf4' if col == 0 else 'white')
            elif label == "Failed":
                cell.set_facecolor('#fef2f2' if col == 0 else 'white')
            else:
                cell.set_facecolor('white')


def _draw_criteria_card(ax, dta, dd, suppression_level, dd_stats, dta_stats):
    """Draw criteria and statistics as a professional card table."""
    ax.set_facecolor('white')
    ax.axis('off')

    rows = [
        ["Criteria", ""],
        ["DD (%)", f"{dd:g}"],
        ["DTA (mm)", f"{dta:g}"],
        ["Threshold (%)", f"{suppression_level:g}"],
    ]
    if dd_stats:
        rows.extend([
            ["DD Statistics", ""],
            ["Mean / Max (%)",
             f"{dd_stats.get('mean', 0):.2f}  /  {dd_stats.get('max', 0):.2f}"],
            ["Std (%)", f"{dd_stats.get('std', 0):.2f}"],
        ])
    if dta_stats:
        rows.extend([
            ["DTA Statistics", ""],
            ["Mean / Max (mm)",
             f"{dta_stats.get('mean', 0):.2f}  /  {dta_stats.get('max', 0):.2f}"],
            ["Std (mm)", f"{dta_stats.get('std', 0):.2f}"],
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Parameter", "Value"],
        colLoc='left', cellLoc='left',
        colWidths=[0.58, 0.42],
        bbox=[0.03, 0.03, 0.94, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    section_headers = {"Criteria", "DD Statistics", "DTA Statistics"}
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(_BORDER)
        cell.set_linewidth(0.6)
        cell.get_text().set_color('black')
        if row == 0:
            cell.set_facecolor(_HEADER_BG)
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            metric = rows[row - 1][0]
            if metric in section_headers:
                cell.set_facecolor(_SECTION_HEADER_BG)
                cell.get_text().set_weight('bold')
                if col == 1:
                    cell.get_text().set_text("")
            else:
                cell.set_facecolor('white')


def _draw_footer(fig, gs_slot):
    """Draw a thin footer strip with software name and date."""
    ax = fig.add_subplot(gs_slot)
    ax.set_facecolor(_FOOTER_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    bg = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle=mpatches.BoxStyle("Square", pad=0),
        facecolor=_FOOTER_BG, edgecolor='none',
        transform=ax.transAxes, zorder=0)
    ax.add_patch(bg)

    ax.text(0.02, 0.5, 'Auto Gamma Analyzer',
            ha='left', va='center', fontsize=9, color=_MUTED_TEXT,
            style='italic', zorder=1)
    ax.text(0.98, 0.5, datetime.now().strftime('%Y-%m-%d %H:%M'),
            ha='right', va='center', fontsize=9, color=_MUTED_TEXT,
            zorder=1)


# ── Main entry point ──────────────────────────────────────────────────

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
    Generates a professional single-page report of the gamma analysis.

    Layout (top to bottom):
      Header  – dark bar with patient / institution (no date)
      Row 1   – DICOM RT Dose  |  MCC Dose (Interpolated)
      Row 2   – Horizontal Profile  |  Vertical Profile
      Row 3   – Gamma Map (left)  |  Pass-rate badge (right)
      Row 4   – DD + DTA side-by-side (left)  |  Criteria & stats card (right)
      Footer  – software name + timestamp
    """
    ext = output_path.rsplit('.', 1)[-1].lower() if '.' in output_path else 'pdf'
    save_format = 'pdf' if ext == 'pdf' else 'jpeg'

    fig = plt.figure(figsize=(12, 18), facecolor='white')

    # ── master grid: 6 rows × 2 cols ──────────────────────────────────
    gs = fig.add_gridspec(
        6, 2,
        height_ratios=[0.20, 1.0, 1.0, 1.0, 1.0, 0.07],
        width_ratios=[1.0, 1.0],
        hspace=0.32, wspace=0.28,
        left=0.07, right=0.95, top=0.97, bottom=0.02,
    )

    # ── Header ────────────────────────────────────────────────────────
    institution, patient_id, patient_name = dicom_handler.get_patient_info()
    _draw_header(fig, gs[0, :], institution, patient_id, patient_name)

    # ── Row 1: Dose distributions ─────────────────────────────────────
    dicom_data = dicom_handler.get_pixel_data()
    dicom_extent = dicom_handler.get_physical_extent()

    ax_dicom = fig.add_subplot(gs[1, 0])
    if dicom_data is not None and dicom_extent is not None:
        im = ax_dicom.imshow(dicom_data, cmap='turbo', extent=dicom_extent,
                             aspect='equal', origin='upper')
        cb = fig.colorbar(im, ax=ax_dicom, label='Dose (Gy)', pad=0.02,
                          shrink=0.92)
        _style_report_colorbar(cb)
    ax_dicom.set_title('DICOM RT Dose', fontsize=13, weight='bold', pad=8)
    ax_dicom.set_xlabel('Position (mm)', fontsize=11)
    ax_dicom.set_ylabel('Position (mm)', fontsize=11)
    ax_dicom.tick_params(labelsize=9)
    _style_report_axis(ax_dicom)

    ax_mcc = fig.add_subplot(gs[1, 1])
    if mcc_interp_data is not None and dicom_extent is not None:
        im = ax_mcc.imshow(mcc_interp_data, cmap='turbo', extent=dicom_extent,
                           aspect='equal', origin='upper')
        cb = fig.colorbar(im, ax=ax_mcc, label='Dose (Gy)', pad=0.02,
                          shrink=0.92)
        _style_report_colorbar(cb)
    ax_mcc.set_title('MCC Dose (Interpolated)', fontsize=13, weight='bold', pad=8)
    ax_mcc.set_xlabel('Position (mm)', fontsize=11)
    ax_mcc.set_ylabel('Position (mm)', fontsize=11)
    ax_mcc.tick_params(labelsize=9)
    _style_report_axis(ax_mcc)

    # ── Row 2: Dose profiles (same height as dose maps) ──────────────
    ax_hprof = fig.add_subplot(gs[2, 0])
    if hor_profile_data:
        ax_hprof.plot(hor_profile_data['phys_coords'],
                      hor_profile_data['dicom_values'],
                      'b-', linewidth=2, label='RT dose')
        if 'mcc_values' in hor_profile_data and 'mcc_phys_coords' in hor_profile_data:
            ax_hprof.plot(hor_profile_data['mcc_phys_coords'],
                          hor_profile_data['mcc_values'],
                          'ro', markersize=5, alpha=0.75, label='MCC dose')
        ax_hprof.set_title('Left-Right Profile (Horizontal)',
                           fontsize=13, weight='bold', pad=8)
        ax_hprof.set_xlabel('Position (mm)', fontsize=11)
        ax_hprof.set_ylabel('Dose (Gy)', fontsize=11)
        ax_hprof.tick_params(labelsize=9)
        legend = ax_hprof.legend(fontsize=9, loc='upper right')
        _style_report_legend(legend)
        ax_hprof.grid(True, alpha=0.25, linestyle='--')
    _style_report_axis(ax_hprof)

    ax_vprof = fig.add_subplot(gs[2, 1])
    if ver_profile_data:
        ax_vprof.plot(ver_profile_data['phys_coords'],
                      ver_profile_data['dicom_values'],
                      'b-', linewidth=2, label='RT dose')
        if 'mcc_values' in ver_profile_data and 'mcc_phys_coords' in ver_profile_data:
            ax_vprof.plot(ver_profile_data['mcc_phys_coords'],
                          ver_profile_data['mcc_values'],
                          'ro', markersize=5, alpha=0.75, label='MCC dose')
        ax_vprof.set_title('In-Out Profile (Vertical)',
                           fontsize=13, weight='bold', pad=8)
        ax_vprof.set_xlabel('Position (mm)', fontsize=11)
        ax_vprof.set_ylabel('Dose (Gy)', fontsize=11)
        ax_vprof.tick_params(labelsize=9)
        legend = ax_vprof.legend(fontsize=9, loc='upper right')
        _style_report_legend(legend)
        ax_vprof.grid(True, alpha=0.25, linestyle='--')
    _style_report_axis(ax_vprof)

    # ── Row 3: Gamma map (left) | Pass-rate badge (right) ────────────
    ax_gamma = fig.add_subplot(gs[3, 0])
    gamma_cmap = get_gamma_colormap()
    gamma_norm = None
    gamma_source = gamma_map_interp if gamma_map_interp is not None else gamma_map
    if gamma_source is not None:
        fv = np.asarray(gamma_source, dtype=float)
        fv = fv[np.isfinite(fv)]
        gamma_norm = get_gamma_norm(np.nanmax(fv) if fv.size else None)

    if gamma_map_interp is not None and dicom_extent is not None:
        im = ax_gamma.imshow(gamma_map_interp, cmap=gamma_cmap, norm=gamma_norm,
                             extent=dicom_extent, aspect='equal', origin='upper',
                             interpolation='bilinear')
        cb = fig.colorbar(im, ax=ax_gamma, label='Gamma Index', pad=0.02,
                          extend='max', shrink=0.92)
        _style_report_colorbar(cb)
        X = np.linspace(dicom_extent[0], dicom_extent[1],
                        gamma_map_interp.shape[1])
        Y = np.linspace(dicom_extent[3], dicom_extent[2],
                        gamma_map_interp.shape[0])
        ax_gamma.contour(X, Y, gamma_map_interp, levels=[1.0],
                         colors='black', linewidths=1.5, alpha=0.8)
    elif gamma_map is not None:
        mcc_extent = mcc_handler.get_physical_extent()
        if mcc_extent is not None:
            im = ax_gamma.imshow(gamma_map, cmap=gamma_cmap, norm=gamma_norm,
                                 extent=mcc_extent, aspect='equal', origin='upper')
            cb = fig.colorbar(im, ax=ax_gamma, label='Gamma Index', pad=0.02,
                              extend='max', shrink=0.92)
            _style_report_colorbar(cb)

    ax_gamma.set_title(f'Gamma Analysis ({dd:g}%/{dta:g}mm)',
                       fontsize=13, weight='bold', pad=8)
    ax_gamma.set_xlabel('Position (mm)', fontsize=11)
    ax_gamma.set_ylabel('Position (mm)', fontsize=11)
    ax_gamma.tick_params(labelsize=9)
    _style_report_axis(ax_gamma)

    # Gamma result table (right of gamma)
    ax_badge = fig.add_subplot(gs[3, 1])
    _draw_gamma_summary_table(ax_badge, gamma_stats)

    # ── Row 4: DD + DTA side-by-side (left) | Criteria card (right) ──
    gs_r4_left = gs[4, 0].subgridspec(1, 2, wspace=0.35)

    # DD map (left sub-panel)
    ax_dd = fig.add_subplot(gs_r4_left[0, 0])
    if dd_map_interp is not None and dicom_extent is not None:
        im = ax_dd.imshow(dd_map_interp, cmap='viridis', extent=dicom_extent,
                          aspect='equal', origin='upper', interpolation='bilinear')
        cb = fig.colorbar(im, ax=ax_dd, label='DD (%)', pad=0.03,
                          extend='both', shrink=0.85)
        _style_report_colorbar(cb)
    elif dd_map is not None:
        mcc_extent = mcc_handler.get_physical_extent()
        if mcc_extent is not None:
            im = ax_dd.imshow(dd_map, cmap='viridis', extent=mcc_extent,
                              aspect='equal', origin='upper')
            cb = fig.colorbar(im, ax=ax_dd, label='DD (%)', pad=0.03,
                              extend='both', shrink=0.85)
            _style_report_colorbar(cb)
    ax_dd.set_title('DD Map', fontsize=11, weight='bold', pad=6)
    ax_dd.set_xlabel('Position (mm)', fontsize=9)
    ax_dd.set_ylabel('Position (mm)', fontsize=9)
    ax_dd.tick_params(labelsize=7)
    _style_report_axis(ax_dd)

    # DTA map (right sub-panel)
    ax_dta = fig.add_subplot(gs_r4_left[0, 1])
    if dta_map_interp is not None and dicom_extent is not None:
        im = ax_dta.imshow(dta_map_interp, cmap='plasma', extent=dicom_extent,
                           aspect='equal', origin='upper', interpolation='bilinear')
        cb = fig.colorbar(im, ax=ax_dta, label='DTA (mm)', pad=0.03,
                          extend='max', shrink=0.85)
        _style_report_colorbar(cb)
    elif dta_map is not None:
        mcc_extent = mcc_handler.get_physical_extent()
        if mcc_extent is not None:
            im = ax_dta.imshow(dta_map, cmap='plasma', extent=mcc_extent,
                               aspect='equal', origin='upper')
            cb = fig.colorbar(im, ax=ax_dta, label='DTA (mm)', pad=0.03,
                              extend='max', shrink=0.85)
            _style_report_colorbar(cb)
    ax_dta.set_title('DTA Map', fontsize=11, weight='bold', pad=6)
    ax_dta.set_xlabel('Position (mm)', fontsize=9)
    ax_dta.set_ylabel('Position (mm)', fontsize=9)
    ax_dta.tick_params(labelsize=7)
    _style_report_axis(ax_dta)

    # Criteria & stats card (right column of row 4)
    ax_criteria = fig.add_subplot(gs[4, 1])
    _draw_criteria_card(ax_criteria, dta, dd, suppression_level,
                        dd_stats, dta_stats)

    # ── Footer ────────────────────────────────────────────────────────
    _draw_footer(fig, gs[5, :])

    # ── Save ──────────────────────────────────────────────────────────
    plt.savefig(output_path, format=save_format, dpi=300,
                facecolor='white', edgecolor='white')
    plt.close(fig)
