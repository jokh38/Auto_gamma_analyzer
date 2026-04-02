import os
import re
import numpy as np
import pydicom
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox, QDialog, QProgressBar, QVBoxLayout
from PyQt5.QtCore import Qt, QSettings

from src.data_manager import DataManager
from src.ui_components import PlotManager
from src.utils import logger, load_app_config, update_app_config_value
from src.load_dcm import load_dcm
from src.load_mcc import load_mcc
from src.file_handlers import DicomFileHandler, MCCFileHandler
from src.analysis import extract_profile_data, perform_gamma_analysis
from src.reporting import generate_report
from src.standard_data_model import ROI_Data


class BatchProgressDialog(QDialog):
    """Small status window shown while auto analysis runs."""

    def __init__(self, total_pairs, parent=None):
        super().__init__(parent)
        self._allow_close = False
        self.setWindowTitle("Auto analysis")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setModal(False)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        self.progress_label = QLabel("Preparing batch analysis...")
        self.beam_label = QLabel("Beam: -")
        self.dicom_label = QLabel("DICOM: -")
        self.mcc_label = QLabel("MCC: -")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max(total_pairs, 1))
        self.progress_bar.setValue(0)

        layout.addWidget(self.progress_label)
        layout.addWidget(self.beam_label)
        layout.addWidget(self.dicom_label)
        layout.addWidget(self.mcc_label)
        layout.addWidget(self.progress_bar)

    def update_status(self, index, total, beam_key, dicom_path, mcc_path):
        self.progress_label.setText(f"Processing {index}/{total}")
        self.beam_label.setText(f"Beam: {beam_key}G")
        self.dicom_label.setText(f"DICOM: {os.path.basename(dicom_path)}")
        self.mcc_label.setText(f"MCC: {os.path.basename(mcc_path)}")
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(index)

    def allow_close(self):
        self._allow_close = True

    def closeEvent(self, event):
        if self._allow_close:
            super().closeEvent(event)
            return
        event.ignore()


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
        self.app_config = load_app_config()
        self.settings = QSettings("AutoGammaAnalyzer", "AutoGammaAnalyzer")

    def _get_initial_dialog_dir(self):
        """Returns the default first-launch directory for file dialogs."""
        if os.name == "nt":
            return "C:\\"
        return os.path.abspath(os.sep)

    def _get_dialog_dir(self, key):
        """Returns a persisted dialog directory or the root directory on first use."""
        saved_dir = self.settings.value(key, "", type=str)
        if saved_dir and os.path.isdir(saved_dir):
            return saved_dir
        return self._get_initial_dialog_dir()

    def _remember_dialog_dir(self, key, file_path):
        """Stores the directory of the selected file for future dialog openings."""
        selected_dir = file_path if os.path.isdir(file_path) else os.path.dirname(file_path)
        if selected_dir and os.path.isdir(selected_dir):
            self.settings.setValue(key, selected_dir)

    @staticmethod
    def _extract_beam_key(filename):
        """Extract the numeric beam identifier immediately before 'G'."""
        base_name = os.path.splitext(os.path.basename(filename))[0]
        match = re.search(r"(\d+)G", base_name, flags=re.IGNORECASE)
        return match.group(1) if match else None

    @staticmethod
    def _beam_sort_key(beam_key):
        try:
            return int(beam_key)
        except (TypeError, ValueError):
            return beam_key

    def _collect_batch_pairs(self, directory):
        """Collect one DICOM/MCC pair per beam number from a directory."""
        grouped_files = {}
        skipped = []

        for entry in sorted(os.listdir(directory)):
            file_path = os.path.join(directory, entry)
            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(entry)[1].lower()
            if extension not in (".dcm", ".mcc"):
                continue

            if extension == ".dcm":
                is_rt_dose, modality = self._is_rt_dose_dicom(file_path)
                if not is_rt_dose:
                    skipped.append(f"{entry}: skipped DICOM modality {modality}")
                    continue

            beam_key = self._extract_beam_key(entry)
            if beam_key is None:
                skipped.append(f"{entry}: beam number not found")
                continue

            pair = grouped_files.setdefault(beam_key, {})
            if extension in pair:
                skipped.append(f"{entry}: duplicate {extension} for beam {beam_key}G")
                pair["duplicate"] = True
                continue

            pair[extension] = file_path

        complete_pairs = {}
        for beam_key, pair in grouped_files.items():
            if pair.get("duplicate"):
                skipped.append(f"{beam_key}G: duplicate beam files")
                continue
            if ".dcm" not in pair or ".mcc" not in pair:
                skipped.append(f"{beam_key}G: incomplete pair")
                continue
            complete_pairs[beam_key] = {"dcm": pair[".dcm"], "mcc": pair[".mcc"]}

        return complete_pairs, skipped

    @staticmethod
    def _is_rt_dose_dicom(file_path):
        """Return whether a DICOM file is an RT Dose object based on its header."""
        try:
            dicom_header = pydicom.dcmread(
                file_path,
                stop_before_pixels=True,
                specific_tags=["Modality"],
            )
            modality = getattr(dicom_header, "Modality", "UNKNOWN")
            return modality == "RTDOSE", modality
        except Exception as exc:
            logger.warning(f"Failed to inspect DICOM header for {file_path}: {exc}")
            return False, "UNREADABLE"

    def _set_batch_ui_enabled(self, enabled):
        """Disable the main app while batch analysis is running."""
        self.main_view.setEnabled(enabled)
        if enabled:
            self.main_view.activateWindow()
            self.main_view.raise_()
        QApplication.processEvents()

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

    def _apply_origin_shift_to_handler(self, handler, delta_x_mm, delta_y_mm):
        """Apply the current origin shift directly to a freshly loaded DICOM handler."""
        if handler is None or not isinstance(handler, DicomFileHandler):
            return
        if handler.phys_x_mesh is None or handler.phys_y_mesh is None or handler.physical_extent is None:
            return
        handler.phys_x_mesh = handler.phys_x_mesh + delta_x_mm
        handler.phys_y_mesh = handler.phys_y_mesh + delta_y_mm
        min_x, max_x, min_y, max_y = handler.physical_extent
        handler.physical_extent = [
            min_x + delta_x_mm,
            max_x + delta_x_mm,
            min_y + delta_y_mm,
            max_y + delta_y_mm,
        ]

    def _create_handler_for_path(self, filename):
        """Load a file into the correct handler without touching the visible UI state."""
        if filename.lower().endswith(".dcm"):
            handler = DicomFileHandler()
            file_type = "DICOM"
        elif filename.lower().endswith(".mcc"):
            handler = MCCFileHandler()
            file_type = "MCC"
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        success, error_msg = handler.open_file(filename)
        if not success:
            raise ValueError(error_msg)

        return handler, file_type

    def _load_batch_pair_handlers(self, dicom_path, mcc_path):
        """Load, normalize, and align one DICOM/MCC pair for headless batch analysis."""
        file_a_handler, file_a_type = self._create_handler_for_path(dicom_path)
        file_b_handler, file_b_type = self._create_handler_for_path(mcc_path)

        if file_a_type != "DICOM" or file_b_type != "MCC":
            raise ValueError("Auto analysis requires one DICOM file and one MCC file per beam.")

        self._apply_handler_normalization(file_a_handler, self.data_manager.file_a_normalization)
        self._apply_handler_normalization(file_b_handler, self.data_manager.file_b_normalization)
        self._apply_origin_shift_to_handler(
            file_a_handler,
            float(self.main_view.dicom_x_spin.value()),
            float(self.main_view.dicom_y_spin.value()),
        )

        if file_a_handler.dose_bounds and hasattr(file_b_handler, "crop_to_bounds"):
            file_b_handler.crop_to_bounds(file_a_handler.dose_bounds)

        return file_a_handler, file_b_handler

    def _resolve_analysis_handlers(self, file_a_handler, file_b_handler):
        """Select reference/evaluation handlers using the same rules as the interactive flow."""
        file_a_is_mcc = isinstance(file_a_handler, MCCFileHandler)
        file_b_is_mcc = isinstance(file_b_handler, MCCFileHandler)

        if file_b_is_mcc and not file_a_is_mcc:
            return file_b_handler, file_a_handler
        if file_a_is_mcc and not file_b_is_mcc:
            return file_a_handler, file_b_handler
        return file_b_handler, file_a_handler

    def _perform_gamma_analysis_for_handlers(self, file_a_handler, file_b_handler):
        """Run gamma analysis without mutating visible UI widgets."""
        dta = self.main_view.dta_spin.value()
        dd = self.main_view.dd_spin.value()
        is_global = self.main_view.gamma_type_combo.currentText() == "Global"
        reference_handler, evaluation_handler = self._resolve_analysis_handlers(file_a_handler, file_b_handler)

        results = perform_gamma_analysis(
            reference_handler=reference_handler,
            evaluation_handler=evaluation_handler,
            dose_percent_threshold=dd,
            distance_mm_threshold=dta,
            global_normalisation=is_global,
            threshold=getattr(reference_handler, "suppression_level", 10),
            save_csv=self.app_config["save_csv"],
            csv_dir=self.app_config["csv_export_path"],
        )
        return results, reference_handler, evaluation_handler

    def _build_default_report_path(self, report_dir, dicom_handler):
        """Build the report filename using the existing naming convention."""
        _, patient_id, _ = dicom_handler.get_patient_info()
        patient_id = patient_id or "unknown"
        dicom_filename = dicom_handler.get_filename()
        dicom_filename_base = os.path.splitext(dicom_filename)[0] if dicom_filename else "file"
        return os.path.join(report_dir, f"report_{patient_id}_{dicom_filename_base}.pdf")

    def _generate_report_for_handlers(
        self,
        output_path,
        file_a_handler,
        file_b_handler,
        gamma_results,
    ):
        """Generate a report for a prepared pair using headless analysis data."""
        (
            gamma_map,
            gamma_stats,
            _phys_extent,
            mcc_interp_data,
            dd_map,
            dta_map,
            dd_stats,
            dta_stats,
            gamma_map_interp,
            dd_map_interp,
            dta_map_interp,
        ) = gamma_results

        ver_profile = extract_profile_data(
            direction="vertical",
            fixed_position=0,
            dicom_handler=file_a_handler,
            mcc_handler=file_b_handler,
        )
        hor_profile = extract_profile_data(
            direction="horizontal",
            fixed_position=0,
            dicom_handler=file_a_handler,
            mcc_handler=file_b_handler,
        )

        generate_report(
            output_path=output_path,
            dicom_handler=file_a_handler,
            mcc_handler=file_b_handler,
            gamma_map=gamma_map,
            gamma_stats=gamma_stats,
            dta=self.main_view.dta_spin.value(),
            dd=self.main_view.dd_spin.value(),
            suppression_level=getattr(file_b_handler, "suppression_level", self.app_config["suppression_level"]),
            ver_profile_data=ver_profile,
            hor_profile_data=hor_profile,
            mcc_interp_data=mcc_interp_data,
            dd_map=dd_map,
            dta_map=dta_map,
            dd_stats=dd_stats,
            dta_stats=dta_stats,
            gamma_map_interp=gamma_map_interp,
            dd_map_interp=dd_map_interp,
            dta_map_interp=dta_map_interp
        )

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

    def clear_data(self):
        """Unload File A and File B and reset the application view state."""
        dm = self.data_manager

        dm.dicom_data = None
        dm.mcc_data = None
        dm.file_a_handler = None
        dm.file_b_handler = None
        dm.dicom_handler = None
        dm.mcc_handler = None
        dm.dicom_roi = None
        dm.mcc_roi = None
        dm.initial_dicom_phys_coords = None
        dm.initial_dicom_pixel_origin = None
        dm.initial_dicom_origin_mm = None
        dm.initial_dicom_handler_meshes = None
        dm.initial_dicom_handler_extent = None
        dm.profile_line = None
        dm.current_profile_data = None

        self._clear_gamma_results()
        self.plot_manager.clear_all_displays()

        self.main_view.dicom_label.setText("File A: None")
        self.main_view.mcc_label.setText("File B: None")
        self.main_view.device_label.setText("Device Type: Not detected")
        self.main_view.dicom_x_spin.setValue(0.0)
        self.main_view.dicom_y_spin.setValue(0.0)

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
            results, reference_handler, evaluation_handler = self._perform_gamma_analysis_for_handlers(
                file_a_handler, file_b_handler
            )
            (
                dm.gamma_map, dm.gamma_stats, dm.phys_extent, dm.mcc_interp_data,
                dm.dd_map, dm.dta_map, dm.dd_stats, dm.dta_stats,
                dm.gamma_map_interp, dm.dd_map_interp, dm.dta_map_interp
            ) = results

            # Update legacy references for backward compatibility
            file_a_is_mcc = isinstance(file_a_handler, MCCFileHandler)
            file_b_is_mcc = isinstance(file_b_handler, MCCFileHandler)
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
        start_dir = self._get_dialog_dir("paths/load_dir")
        filename, _ = QFileDialog.getOpenFileName(
            self.main_view, "Open File A (Top)", start_dir,
            "All Supported Files (*.dcm *.mcc);;DICOM Files (*.dcm);;MCC Files (*.mcc);;All Files (*)",
            options=options
        )
        if not filename:
            return
        self._remember_dialog_dir("paths/load_dir", filename)

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
        start_dir = self._get_dialog_dir("paths/load_dir")
        filename, _ = QFileDialog.getOpenFileName(
            self.main_view, "Open File B (Bottom)", start_dir,
            "All Supported Files (*.dcm *.mcc);;DICOM Files (*.dcm);;MCC Files (*.mcc);;All Files (*)",
            options=options
        )
        if not filename:
            return
        self._remember_dialog_dir("paths/load_dir", filename)

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

    def update_suppression_level(self):
        """Persist the dose suppression level and refresh gamma analysis."""
        new_value = int(self.main_view.suppression_spin.value())
        self.app_config["suppression_level"] = new_value
        try:
            update_app_config_value("suppression_level", new_value)
        except Exception as exc:
            QMessageBox.warning(
                self.main_view,
                "Warning",
                f"Could not save dose suppression to config.yaml: {exc}",
            )

        dm = self.data_manager
        for handler in {
            dm.file_a_handler,
            dm.file_b_handler,
            dm.dicom_handler,
            dm.mcc_handler,
        }:
            if handler is not None and hasattr(handler, "suppression_level"):
                handler.suppression_level = new_value

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
            report_dir = self._get_dialog_dir("paths/report_dir")

            default_path = self._build_default_report_path(report_dir, dm.file_a_handler)
            output_path, _ = QFileDialog.getSaveFileName(self.main_view, "Save Report", default_path, "PDF Document (*.pdf);;JPEG Image (*.jpg *.jpeg)")
            if not output_path: return
            self._remember_dialog_dir("paths/report_dir", output_path)
            self._generate_report_for_handlers(
                output_path=output_path,
                file_a_handler=dm.file_a_handler,
                file_b_handler=dm.file_b_handler,
                gamma_results=(
                    dm.gamma_map,
                    dm.gamma_stats,
                    dm.phys_extent,
                    dm.mcc_interp_data,
                    dm.dd_map,
                    dm.dta_map,
                    dm.dd_stats,
                    dm.dta_stats,
                    dm.gamma_map_interp,
                    dm.dd_map_interp,
                    dm.dta_map_interp,
                ),
            )
            QMessageBox.information(self.main_view, "Success", f"Report saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self.main_view, "Error", f"Failed to generate report: {e}")
            logger.error(f"Report generation error: {e}", exc_info=True)

    def auto_analysis(self):
        """Batch-create reports for all complete DICOM/MCC beam pairs in a directory."""
        start_dir = self._get_dialog_dir("paths/auto_analysis_dir")
        selected_dir = QFileDialog.getExistingDirectory(
            self.main_view,
            "Select Directory for Auto analysis",
            start_dir,
        )
        if not selected_dir:
            return

        self._remember_dialog_dir("paths/auto_analysis_dir", selected_dir)
        self._remember_dialog_dir("paths/report_dir", selected_dir)

        complete_pairs, skipped = self._collect_batch_pairs(selected_dir)
        if not complete_pairs:
            details = "\n".join(skipped) if skipped else "No matching DICOM/MCC beam pairs were found."
            QMessageBox.warning(self.main_view, "Auto analysis", details)
            return

        sorted_pairs = sorted(complete_pairs.items(), key=lambda item: self._beam_sort_key(item[0]))
        progress_dialog = BatchProgressDialog(len(sorted_pairs))
        progress_dialog.show()
        progress_dialog.raise_()
        QApplication.processEvents()

        success_paths = []
        failures = []

        self._set_batch_ui_enabled(False)
        try:
            for index, (beam_key, pair) in enumerate(sorted_pairs, start=1):
                progress_dialog.update_status(index, len(sorted_pairs), beam_key, pair["dcm"], pair["mcc"])
                QApplication.processEvents()

                try:
                    file_a_handler, file_b_handler = self._load_batch_pair_handlers(pair["dcm"], pair["mcc"])
                    gamma_results, _reference_handler, _evaluation_handler = self._perform_gamma_analysis_for_handlers(
                        file_a_handler, file_b_handler
                    )
                    output_path = self._build_default_report_path(selected_dir, file_a_handler)
                    self._generate_report_for_handlers(output_path, file_a_handler, file_b_handler, gamma_results)
                    success_paths.append(output_path)
                except Exception as exc:
                    logger.error(f"Auto analysis failed for beam {beam_key}G: {exc}", exc_info=True)
                    failures.append(f"{beam_key}G: {exc}")
        finally:
            self._set_batch_ui_enabled(True)
            progress_dialog.allow_close()
            progress_dialog.close()

        summary_lines = [
            f"Created {len(success_paths)} report(s).",
        ]
        if skipped:
            summary_lines.append(f"Skipped {len(skipped)} item(s).")
        if failures:
            summary_lines.append(f"Failed {len(failures)} beam(s).")

        details = []
        if success_paths:
            details.append("Reports:")
            details.extend(success_paths)
        if skipped:
            details.append("Skipped:")
            details.extend(skipped)
        if failures:
            details.append("Failed:")
            details.extend(failures)

        message = "\n".join(summary_lines)
        if details:
            message = f"{message}\n\n" + "\n".join(details)

        QMessageBox.information(self.main_view, "Auto analysis", message)
