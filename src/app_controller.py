import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

from src.data_manager import DataManager
from src.ui_components import PlotManager
from src.utils import logger
from src.load_dcm import load_dcm
from src.load_mcc import load_mcc
from src.file_handlers import DicomFileHandler, MCCFileHandler
from src.analysis import perform_gamma_analysis
from src.reporting import generate_report
from src.standard_data_model import ROI_Data

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

    def _set_file_label(self, label_widget, prefix, filename):
        """Keep file labels from changing splitter widths when long names are loaded."""
        try:
            metrics = label_widget.fontMetrics()
            width = label_widget.width()
            available_width = width if isinstance(width, (int, float)) else 220
            available_width = max(int(available_width), 220)
            elided_name = metrics.elidedText(filename, Qt.ElideRight, available_width)
            if not isinstance(elided_name, str):
                elided_name = filename
        except Exception:
            elided_name = filename
        label_widget.setText(f"{prefix}: {elided_name}")

    def _apply_handler_normalization(self, handler, factor):
        """Applies the current normalization factor to a loaded handler."""
        if handler and hasattr(handler, "set_normalization_factor"):
            handler.set_normalization_factor(factor)

    def _clear_gamma_results(self):
        """Clears gamma outputs when upstream dose data changes."""
        dm = self.data_manager
        dm.gamma_map = None
        dm.gamma_stats = None
        dm.phys_extent = None
        dm.mcc_interp_data = None
        dm.dd_map = None
        dm.dta_map = None
        dm.dd_stats = None
        dm.dta_stats = None
        dm.gamma_map_interp = None
        dm.dd_map_interp = None
        dm.dta_map_interp = None
        self.main_view.generate_report_btn.setEnabled(False)
        self.plot_manager.draw_gamma_map()

    def _can_run_gamma_analysis(self):
        """Returns whether both files needed for gamma analysis are available."""
        dm = self.data_manager
        file_a_handler = dm.file_a_handler or dm.dicom_handler or dm.mcc_handler
        file_b_handler = dm.file_b_handler or (
            dm.mcc_handler if dm.mcc_handler is not file_a_handler else dm.dicom_handler
        )
        return file_a_handler is not None and file_b_handler is not None

    def _auto_run_gamma_analysis_if_ready(self):
        """Recompute gamma automatically when both loaded dose files are available."""
        if self._can_run_gamma_analysis():
            self.run_gamma_analysis()
        else:
            self._clear_gamma_results()

    def update_normalization(self, side, factor):
        """Updates the A/B normalization factor and refreshes dependent views."""
        dm = self.data_manager

        if side == "A":
            dm.file_a_normalization = float(factor)
            self._apply_handler_normalization(dm.file_a_handler, dm.file_a_normalization)
        elif side == "B":
            dm.file_b_normalization = float(factor)
            self._apply_handler_normalization(dm.file_b_handler, dm.file_b_normalization)
        else:
            raise ValueError(f"Unsupported normalization side: {side}")

        self.plot_manager.redraw_all_images()
        if dm.profile_line is not None:
            self.generate_and_draw_profile()
        self._auto_run_gamma_analysis_if_ready()

    def _apply_dicom_shift(self, delta_x_mm, delta_y_mm):
        """Applies a relative physical shift to the loaded DICOM datasets."""
        dm = self.data_manager

        if dm.dicom_data is not None and dm.initial_dicom_phys_coords is not None:
            dm.dicom_data.x_coords = dm.initial_dicom_phys_coords[0] + delta_x_mm
            dm.dicom_data.y_coords = dm.initial_dicom_phys_coords[1] + delta_y_mm

        if (
            dm.dicom_handler is not None
            and dm.initial_dicom_handler_meshes is not None
            and dm.initial_dicom_handler_extent is not None
        ):
            base_x_mesh, base_y_mesh = dm.initial_dicom_handler_meshes
            dm.dicom_handler.phys_x_mesh = base_x_mesh + delta_x_mm
            dm.dicom_handler.phys_y_mesh = base_y_mesh + delta_y_mm
            min_x, max_x, min_y, max_y = dm.initial_dicom_handler_extent
            dm.dicom_handler.physical_extent = [
                min_x + delta_x_mm,
                max_x + delta_x_mm,
                min_y + delta_y_mm,
                max_y + delta_y_mm,
            ]

    def run_gamma_analysis(self):
        """
        Executes the gamma analysis using parameters from the UI.
        Automatically determines reference and evaluation handlers based on file types.
        """
        dm = self.data_manager
        view = self.main_view

        # Check if both files are loaded
        file_a_handler = dm.file_a_handler or dm.dicom_handler or dm.mcc_handler
        file_b_handler = dm.file_b_handler or (
            dm.mcc_handler if dm.mcc_handler is not file_a_handler else dm.dicom_handler
        )
        if not file_a_handler or not file_b_handler:
            QMessageBox.warning(view, "Warning", "Both File A and File B must be loaded before running gamma analysis.")
            return

        try:
            dta = view.dta_spin.value()
            dd = view.dd_spin.value()
            is_global = view.gamma_type_combo.currentText() == "Global"

            # Determine reference and evaluation handlers
            # Priority: MCC as reference (measurement), DICOM as evaluation (plan)
            # If both are same type, File B is reference, File A is evaluation
            file_a_is_mcc = isinstance(file_a_handler, MCCFileHandler)
            file_b_is_mcc = isinstance(file_b_handler, MCCFileHandler)

            if file_b_is_mcc and not file_a_is_mcc:
                # Standard case: File B (MCC) is reference, File A (DICOM) is evaluation
                reference_handler = file_b_handler
                evaluation_handler = file_a_handler
            elif file_a_is_mcc and not file_b_is_mcc:
                # Reversed case: File A (MCC) is reference, File B (DICOM) is evaluation
                reference_handler = file_a_handler
                evaluation_handler = file_b_handler
            else:
                # Both are same type: File B is reference, File A is evaluation
                reference_handler = file_b_handler
                evaluation_handler = file_a_handler

            results = perform_gamma_analysis(
                reference_handler=reference_handler,
                evaluation_handler=evaluation_handler,
                dose_percent_threshold=dd,
                distance_mm_threshold=dta,
                global_normalisation=is_global,
                threshold=getattr(reference_handler, "suppression_level", 10)
            )
            (
                dm.gamma_map, dm.gamma_stats, dm.phys_extent, dm.mcc_interp_data,
                dm.dd_map, dm.dta_map, dm.dd_stats, dm.dta_stats,
                dm.gamma_map_interp, dm.dd_map_interp, dm.dta_map_interp
            ) = results

            # Update legacy references for backward compatibility
            if file_b_is_mcc:
                dm.mcc_handler = dm.file_b_handler
            elif file_a_is_mcc:
                dm.mcc_handler = dm.file_a_handler
            else:
                dm.mcc_handler = dm.file_b_handler

            if not file_a_is_mcc:
                dm.dicom_handler = dm.file_a_handler
            elif not file_b_is_mcc:
                dm.dicom_handler = dm.file_b_handler

            self.plot_manager.draw_gamma_map()
            if dm.profile_line is not None:
                self.generate_and_draw_profile()

            if 'pass_rate' in dm.gamma_stats:
                view.generate_report_btn.setEnabled(True)
            else:
                QMessageBox.warning(view, "Warning", "No valid gamma results.")
                view.generate_report_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(view, "Error", f"Gamma analysis failed: {e}")
            logger.error(f"Gamma analysis error: {e}", exc_info=True)
            view.generate_report_btn.setEnabled(False)

    def _extract_roi_from_data(self, data, threshold_percent=1.0):
        """
        Identifies a Region of Interest (ROI) based on a dose threshold
        and extracts all relevant data into an ROI_Data object.
        """
        if data is None:
            return None

        max_dose = np.max(data.data_grid)
        if max_dose <= 0:
            return None

        # Determine the ROI based on the threshold
        threshold_val = (threshold_percent / 100.0) * max_dose
        mask = data.data_grid >= threshold_val
        if not np.any(mask):
            return None

        # Find the bounding box of the mask in terms of indices
        rows, cols = np.where(mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Extract the data for the ROI
        y_indices = np.arange(min_row, max_row + 1)
        x_indices = np.arange(min_col, max_col + 1)

        roi_grid = data.data_grid[min_row:max_row+1, min_col:max_col+1]
        y_coords = data.y_coords[y_indices]
        x_coords = data.x_coords[x_indices]

        # Calculate physical extent for plotting, considering pixel edges
        dx = (x_coords[1] - x_coords[0]) / 2.0 if len(x_coords) > 1 else 0.5
        dy = (y_coords[1] - y_coords[0]) / 2.0 if len(y_coords) > 1 else 0.5
        physical_extent = [
            x_coords[0] - dx, x_coords[-1] + dx,
            y_coords[0] - dy, y_coords[-1] + dy
        ]

        # Create and return the ROI_Data object
        return ROI_Data(
            dose_grid=roi_grid,
            x_coords=x_coords,
            y_coords=y_coords,
            x_indices=x_indices,
            y_indices=y_indices,
            physical_extent=physical_extent,
            source_metadata=data.metadata.copy()
        )

    def load_file_a(self):
        """Load File A (Top display) - supports both DCM and MCC files."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(
            self.main_view, "Open File A (Top)", "./",
            "All Supported Files (*.dcm *.mcc);;DICOM Files (*.dcm);;MCC Files (*.mcc);;All Files (*)",
            options=options
        )
        if not filename:
            return

        try:
            # Determine file type and create appropriate handler
            if filename.lower().endswith('.dcm'):
                handler = DicomFileHandler()
                file_type = "DICOM"
            elif filename.lower().endswith('.mcc'):
                handler = MCCFileHandler()
                file_type = "MCC"
            else:
                raise ValueError("Unsupported file type. Please select a .dcm or .mcc file.")

            success, error_msg = handler.open_file(filename)
            if not success:
                if file_type == "DICOM":
                    dicom_data = load_dcm(filename)
                    self.data_manager.dicom_data = dicom_data
                    self.data_manager.dicom_roi = self._extract_roi_from_data(dicom_data)
                    self.plot_manager.redraw_all_images()
                    self._set_file_label(
                        self.main_view.dicom_label,
                        f"File A ({file_type})",
                        os.path.basename(filename)
                    )
                    return
                raise Exception(error_msg)

            # Store handler in data manager
            self.data_manager.file_a_handler = handler
            self._apply_handler_normalization(handler, self.data_manager.file_a_normalization)

            # For backward compatibility with existing code
            if file_type == "DICOM":
                self.data_manager.dicom_handler = handler
                dicom_data = load_dcm(filename)
                self.data_manager.dicom_data = dicom_data
                self.data_manager.dicom_roi = self._extract_roi_from_data(dicom_data)
                self.data_manager.initial_dicom_phys_coords = (dicom_data.x_coords.copy(), dicom_data.y_coords.copy())
                self.data_manager.initial_dicom_handler_meshes = (
                    handler.phys_x_mesh.copy(),
                    handler.phys_y_mesh.copy(),
                )
                self.data_manager.initial_dicom_handler_extent = list(handler.get_physical_extent())
                pos_x, _, pos_z = dicom_data.metadata['image_position_patient']
                spacing_x, spacing_y = dicom_data.metadata['pixel_spacing']
                self.data_manager.initial_dicom_origin_mm = (float(pos_x), float(pos_z))
                pixel_origin_x = int(round(pos_x / spacing_x))
                pixel_origin_y = int(round(pos_z / spacing_y))
                self.data_manager.initial_dicom_pixel_origin = (pixel_origin_x, pixel_origin_y)
                self.main_view.dicom_x_spin.setValue(0.0)
                self.main_view.dicom_y_spin.setValue(0.0)

            self.plot_manager.redraw_all_images()
            self._set_file_label(
                self.main_view.dicom_label,
                f"File A ({file_type})",
                os.path.basename(filename)
            )

            if file_type == "MCC":
                self.main_view.device_label.setText(f"Device Type: {handler.get_device_name()}")

            # Auto-generate profile if both files are loaded
            if self.data_manager.file_b_handler is not None:
                self.set_default_profile_and_generate()
                self._auto_run_gamma_analysis_if_ready()

        except Exception as e:
            QMessageBox.critical(self.main_view, "Error", f"Failed to load File A: {e}")
            logger.error(f"File A load error: {e}", exc_info=True)

    def load_dicom_file(self):
        """Legacy method - redirects to load_file_a for backward compatibility."""
        self.load_file_a()

    def load_file_b(self):
        """Load File B (Bottom display) - supports both DCM and MCC files."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(
            self.main_view, "Open File B (Bottom)", "./",
            "All Supported Files (*.dcm *.mcc);;DICOM Files (*.dcm);;MCC Files (*.mcc);;All Files (*)",
            options=options
        )
        if not filename:
            return

        try:
            # Determine file type and create appropriate handler
            if filename.lower().endswith('.dcm'):
                handler = DicomFileHandler()
                file_type = "DICOM"
            elif filename.lower().endswith('.mcc'):
                handler = MCCFileHandler()
                file_type = "MCC"
            else:
                raise ValueError("Unsupported file type. Please select a .dcm or .mcc file.")

            success, error_msg = handler.open_file(filename)
            if not success:
                if file_type == "MCC":
                    mcc_data = load_mcc(filename)
                    self.data_manager.mcc_data = mcc_data
                    self.data_manager.mcc_roi = self._extract_roi_from_data(mcc_data)
                    self.plot_manager.redraw_all_images()
                    self._set_file_label(
                        self.main_view.mcc_label,
                        f"File B ({file_type})",
                        os.path.basename(filename)
                    )
                    return
                raise Exception(error_msg)

            # Store handler in data manager
            self.data_manager.file_b_handler = handler
            self._apply_handler_normalization(handler, self.data_manager.file_b_normalization)

            # Crop to match File A bounds when the File B handler supports it.
            if self.data_manager.file_a_handler and hasattr(self.data_manager.file_a_handler, 'dose_bounds'):
                if self.data_manager.file_a_handler.dose_bounds and hasattr(handler, 'crop_to_bounds'):
                    handler.crop_to_bounds(self.data_manager.file_a_handler.dose_bounds)

            # For backward compatibility with existing code
            # Use load_mcc() as the primary loader for standardized data processing
            if file_type == "MCC":
                self.data_manager.mcc_handler = handler  # Keep handler for legacy methods
                mcc_data = load_mcc(filename)  # Primary loader with standardized interpolation (fill_value=0.0)
                self.data_manager.mcc_data = mcc_data
                self.data_manager.mcc_roi = self._extract_roi_from_data(mcc_data)
                self.main_view.device_label.setText(f"Device Type: {handler.get_device_name()}")

            self.plot_manager.redraw_all_images()
            self._set_file_label(
                self.main_view.mcc_label,
                f"File B ({file_type})",
                os.path.basename(filename)
            )

            # Auto-generate profile if both files are loaded
            if self.data_manager.file_a_handler is not None:
                self.set_default_profile_and_generate()
                self._auto_run_gamma_analysis_if_ready()

        except Exception as e:
            QMessageBox.critical(self.main_view, "Error", f"Failed to load File B: {e}")
            logger.error(f"File B load error: {e}", exc_info=True)

    def load_measurement_file(self):
        """Legacy method - redirects to load_file_b for backward compatibility."""
        self.load_file_b()

    def set_default_profile_and_generate(self):
        self.set_profile_direction(self.main_view.profile_direction)
        self.plot_manager.redraw_all_images()
        self.generate_and_draw_profile()

    def update_origin(self):
        dm = self.data_manager
        if not dm.dicom_data or not dm.initial_dicom_phys_coords:
            return

        delta_x_mm = float(self.main_view.dicom_x_spin.value())
        delta_y_mm = float(self.main_view.dicom_y_spin.value())
        self._apply_dicom_shift(delta_x_mm, delta_y_mm)

        self.plot_manager.redraw_all_images()
        if dm.profile_line is not None:
            self.generate_and_draw_profile()
        self._auto_run_gamma_analysis_if_ready()

    def update_gamma_parameters(self):
        """Refresh gamma analysis after any criterion change."""
        self._auto_run_gamma_analysis_if_ready()

    def set_profile_direction(self, direction):
        self.main_view.profile_direction = direction
        self.main_view.vertical_btn.setChecked(direction == "vertical")
        self.main_view.horizontal_btn.setChecked(direction == "horizontal")
        if self.plot_manager.profile_table is not None:
            self.plot_manager.profile_table.set_profile_direction(direction)
        if self.data_manager.dicom_data:
            dm = self.data_manager
            dm.profile_line = {"type": direction, "x": 0} if direction == "vertical" else {"type": "horizontal", "y": 0}
            self.plot_manager.redraw_all_images()
            if dm.gamma_stats is not None:
                self.plot_manager.draw_gamma_map()
            self.generate_and_draw_profile()

    def on_dicom_click_handler(self, event):
        self.plot_manager.handle_image_click(event, self.main_view.profile_direction, source="A")

    def on_mcc_click_handler(self, event):
        self.plot_manager.handle_image_click(event, self.main_view.profile_direction, source="B")

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

        dm = self.data_manager

        # Check if handlers are available
        if not dm.file_a_handler or not dm.file_b_handler:
            QMessageBox.warning(self.main_view, "Warning", "Both File A and File B must be available to generate report.")
            return

        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            report_dir = os.path.join(base_dir, 'Report')
            os.makedirs(report_dir, exist_ok=True)

            # Use File A as the primary report source.
            institution, patient_id, patient_name = dm.file_a_handler.get_patient_info()
            dicom_filename = dm.file_a_handler.get_filename()
            dicom_filename_base = os.path.splitext(dicom_filename)[0] if dicom_filename else 'file'

            default_path = os.path.join(report_dir, f"report_{patient_id}_{dicom_filename_base}.jpg")
            output_path, _ = QFileDialog.getSaveFileName(self.main_view, "Save Report", default_path, "JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf)")
            if not output_path: return

            original_profile_line = dm.profile_line
            dm.profile_line = {"type": "vertical", "x": 0}
            ver_profile = self.plot_manager.generate_profile_data()
            dm.profile_line = {"type": "horizontal", "y": 0}
            hor_profile = self.plot_manager.generate_profile_data()
            dm.profile_line = original_profile_line

            generate_report(
                output_path=output_path,
                dicom_handler=dm.file_a_handler,
                mcc_handler=dm.file_b_handler,
                gamma_map=dm.gamma_map,
                gamma_stats=dm.gamma_stats,
                dta=self.main_view.dta_spin.value(),
                dd=self.main_view.dd_spin.value(),
                suppression_level=10,
                ver_profile_data=ver_profile,
                hor_profile_data=hor_profile,
                mcc_interp_data=dm.mcc_interp_data,
                dd_map=dm.dd_map,
                dta_map=dm.dta_map,
                dd_stats=dm.dd_stats,
                dta_stats=dm.dta_stats,
                gamma_map_interp=dm.gamma_map_interp,
                dd_map_interp=dm.dd_map_interp,
                dta_map_interp=dm.dta_map_interp
            )
            QMessageBox.information(self.main_view, "Success", f"Report saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self.main_view, "Error", f"Failed to generate report: {e}")
            logger.error(f"Report generation error: {e}", exc_info=True)
