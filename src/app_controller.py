import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from src.data_manager import DataManager
from src.ui_components import PlotManager
from src.utils import logger
from src.load_dcm import load_dcm
from src.load_mcc import load_mcc
from src.analysis import perform_gamma_analysis
from src.reporting import generate_report

class AppController:
    """
    Handles all application logic and acts as the controller in the MVC pattern.
    It responds to user inputs from the view and updates the model (DataManager)
    and the view (via PlotManager).
    """
    def __init__(self, main_view, data_manager: DataManager, plot_manager: PlotManager):
        self.main_view = main_view
        self.data_manager = data_manager
        self.plot_manager = plot_manager

    def run_gamma_analysis(self):
        """
        Executes the gamma analysis using parameters from the UI.
        """
        dm = self.data_manager
        view = self.main_view

        if not dm.dicom_data or not dm.mcc_data:
            QMessageBox.warning(view, "Warning", "Both DICOM and Measurement data must be loaded.")
            return

        try:
            dta = view.dta_spin.value()
            dd = view.dd_spin.value()
            is_global = view.gamma_type_combo.currentText() == "Global"

            results = perform_gamma_analysis(dm.mcc_data, dm.dicom_data, dd, dta, is_global)
            (
                dm.gamma_map, dm.gamma_stats, dm.phys_extent, dm.mcc_interp_data,
                dm.dd_map, dm.dta_map, dm.dd_stats, dm.dta_stats
            ) = results

            self.plot_manager.draw_gamma_map()

            if 'pass_rate' in dm.gamma_stats:
                stats_text = f"Gamma Stats: Pass = {dm.gamma_stats['pass_rate']:.2f}% | Mean = {dm.gamma_stats['mean']:.3f} | Max = {dm.gamma_stats['max']:.3f}"
                view.gamma_stats_label.setText(stats_text)
                view.generate_report_btn.setEnabled(True)
            else:
                QMessageBox.warning(view, "Warning", "No valid gamma results.")
                view.generate_report_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(view, "Error", f"Gamma analysis failed: {e}")
            logger.error(f"Gamma analysis error: {e}", exc_info=True)
            view.generate_report_btn.setEnabled(False)

    def _calculate_dose_bounds(self, data, threshold_percent=1, margin_mm=20):
        if data is None: return None
        max_dose = np.max(data.data_grid)
        if max_dose <= 0: return None
        threshold_val = (threshold_percent / 100.0) * max_dose
        mask = data.data_grid >= threshold_val
        if not np.any(mask): return None
        rows, cols = np.where(mask)
        min_phys_x = data.x_coords[cols.min()] - margin_mm
        max_phys_x = data.x_coords[cols.max()] + margin_mm
        min_phys_y = data.y_coords[rows.min()] - margin_mm
        max_phys_y = data.y_coords[rows.max()] + margin_mm
        return {'min_x': min_phys_x, 'max_x': max_phys_x, 'min_y': min_phys_y, 'max_y': max_phys_y}

    def load_dicom_file(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self.main_view, "Open DICOM RT Dose File", "./", "DICOM Files (*.dcm);;All Files (*)", options=options)
        if not filename: return
        try:
            self.data_manager.dicom_data = load_dcm(filename)
            self.data_manager.dose_bounds = self._calculate_dose_bounds(self.data_manager.dicom_data)
            self.data_manager.initial_dicom_phys_coords = (self.data_manager.dicom_data.x_coords.copy(), self.data_manager.dicom_data.y_coords.copy())
            pos_x, pos_y, _ = self.data_manager.dicom_data.metadata['image_position_patient']
            spacing_x, spacing_y = self.data_manager.dicom_data.metadata['pixel_spacing']
            pixel_origin_x = int(round(pos_x / spacing_x))
            pixel_origin_y = int(round(pos_y / spacing_y))
            self.data_manager.initial_dicom_pixel_origin = (pixel_origin_x, pixel_origin_y)
            self.main_view.dicom_x_spin.setValue(pixel_origin_x)
            self.main_view.dicom_y_spin.setValue(pixel_origin_y)
            self.plot_manager.redraw_all_images()
            self.main_view.dicom_label.setText(f"DICOM RT Dose: {os.path.basename(filename)}")
            self.main_view.origin_label.setText(f"DICOM Physical Origin: ({pos_x:.2f}, {pos_y:.2f}) mm")
            if self.data_manager.mcc_data is not None:
                self.set_default_profile_and_generate()
        except Exception as e:
            QMessageBox.critical(self.main_view, "Error", f"Failed to load DICOM file: {e}")
            logger.error(f"DICOM load error: {e}", exc_info=True)

    def load_measurement_file(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self.main_view, "Open Measurement File", "./", "MCC Files (*.mcc);;All Files (*)", options=options)
        if not filename: return
        try:
            if filename.lower().endswith('.mcc'):
                self.data_manager.mcc_data = load_mcc(filename)
                self.plot_manager.redraw_all_images()
                meta = self.data_manager.mcc_data.metadata
                self.main_view.device_label.setText(f"Device Type: {meta['device']}")
                self.main_view.mcc_label.setText(f"MCC File: {os.path.basename(filename)}")
            else:
                QMessageBox.warning(self.main_view, "Warning", f"Unsupported file type: {os.path.basename(filename)}")
                return
            if self.data_manager.dicom_data is not None:
                self.set_default_profile_and_generate()
        except Exception as e:
            QMessageBox.critical(self.main_view, "Error", f"Failed to load measurement file: {e}")
            logger.error(f"Measurement file load error: {e}", exc_info=True)

    def set_default_profile_and_generate(self):
        self.set_profile_direction(self.main_view.profile_direction)
        self.plot_manager.redraw_all_images()
        self.generate_and_draw_profile()

    def update_origin(self):
        dm = self.data_manager
        if not dm.dicom_data or not dm.initial_dicom_phys_coords: return
        spacing_x, spacing_y = dm.dicom_data.metadata['pixel_spacing']
        pixel_offset_x = self.main_view.dicom_x_spin.value() - dm.initial_dicom_pixel_origin[0]
        pixel_offset_y = self.main_view.dicom_y_spin.value() - dm.initial_dicom_pixel_origin[1]
        phys_offset_x = pixel_offset_x * spacing_x
        phys_offset_y = pixel_offset_y * spacing_y
        dm.dicom_data.x_coords = dm.initial_dicom_phys_coords[0] + phys_offset_x
        dm.dicom_data.y_coords = dm.initial_dicom_phys_coords[1] + phys_offset_y
        self.plot_manager.redraw_all_images()
        self.generate_and_draw_profile()

    def set_profile_direction(self, direction):
        self.main_view.profile_direction = direction
        self.main_view.vertical_btn.setChecked(direction == "vertical")
        self.main_view.horizontal_btn.setChecked(direction == "horizontal")
        if self.data_manager.dicom_data:
            dm = self.data_manager
            dm.profile_line = {"type": direction, "x": 0} if direction == "vertical" else {"type": "horizontal", "y": 0}
            self.plot_manager.redraw_all_images()
            self.generate_and_draw_profile()

    def on_dicom_click_handler(self, event):
        self.plot_manager.handle_dicom_click(event, self.main_view.profile_direction)

    def generate_and_draw_profile(self):
        try:
            profile_data = self.plot_manager.generate_profile_data()
            self.plot_manager.draw_profile(profile_data, self.main_view.profile_direction)
        except Exception as e:
            logger.error(f"Profile generation error: {e}", exc_info=True)
            QMessageBox.warning(self.main_view, "Warning", f"Could not generate profile: {e}")

    def generate_report(self):
        if self.data_manager.gamma_stats is None:
            QMessageBox.warning(self.main_view, "Warning", "Run gamma analysis first.")
            return
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            report_dir = os.path.join(base_dir, 'Report')
            os.makedirs(report_dir, exist_ok=True)
            patient_id = self.data_manager.dicom_data.metadata.get('patient_id', 'Unknown')
            dicom_filename_base = os.path.splitext(os.path.basename(self.data_manager.dicom_data.metadata.get('filename', 'file')))[0]
            default_path = os.path.join(report_dir, f"report_{patient_id}_{dicom_filename_base}.jpg")
            output_path, _ = QFileDialog.getSaveFileName(self.main_view, "Save Report", default_path, "JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf)")
            if not output_path: return

            original_profile_line = self.data_manager.profile_line
            self.data_manager.profile_line = {"type": "vertical", "x": 0}
            ver_profile = self.plot_manager.generate_profile_data()
            self.data_manager.profile_line = {"type": "horizontal", "y": 0}
            hor_profile = self.plot_manager.generate_profile_data()
            self.data_manager.profile_line = original_profile_line

            generate_report(
                output_path=output_path, dicom_data=self.data_manager.dicom_data, mcc_data=self.data_manager.mcc_data,
                gamma_map=self.data_manager.gamma_map, gamma_stats=self.data_manager.gamma_stats,
                dta=self.main_view.dta_spin.value(), dd=self.main_view.dd_spin.value(), suppression_level=10,
                ver_profile_data=ver_profile, hor_profile_data=hor_profile,
                mcc_interp_data=self.data_manager.mcc_interp_data, dd_stats=self.data_manager.dd_stats, dta_stats=self.data_manager.dta_stats,
                dose_bounds=self.data_manager.dose_bounds
            )
            QMessageBox.information(self.main_view, "Success", f"Report saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self.main_view, "Error", f"Failed to generate report: {e}")
            logger.error(f"Report generation error: {e}", exc_info=True)
