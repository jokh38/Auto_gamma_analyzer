import sys
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSpinBox, QComboBox, QGridLayout, 
                             QScrollArea, QGroupBox, QMessageBox)
from typing import Optional

from src.utils import logger
from src.standard_data_model import StandardDoseData
from src.loaders import load_dcm, load_mcc
from src.ui_components import MatplotlibCanvas, ProfileDataTable, draw_image
from src.analysis import extract_profile_data, perform_gamma_analysis
from src.reporting import generate_report

class GammaAnalysisApp(QMainWindow):
    """2D 감마 분석 애플리케이션"""
    def __init__(self):
        super().__init__()
        
        # 변수 초기화
        self.dicom_data: Optional[StandardDoseData] = None
        self.mcc_data: Optional[StandardDoseData] = None

        # 원점 조정을 위한 원본 좌표 저장
        self.initial_dicom_phys_coords = None
        self.initial_dicom_pixel_origin = None

        self.profile_line = None
        self.current_profile_data = None
        self.profile_direction = "vertical"
        self.dose_bounds = None
        
        # 분석 결과 저장을 위한 변수
        self.gamma_map = None
        self.gamma_stats = None
        self.phys_extent = None
        self.mcc_interp_data = None
        self.dd_map = None
        self.dta_map = None
        self.dd_stats = None
        self.dta_stats = None
        
        # UI 설정
        self.init_ui()

    def init_ui(self):
        """UI 컴포넌트 초기화"""
        self.setWindowTitle('2D Gamma Analysis')
        self.setGeometry(100, 100, 1000, 1000)
                
        # 메인 위젯 및 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 상단 제어 패널
        control_panel = QWidget()
        control_layout = QGridLayout(control_panel)
        
        # 파일 로드 버튼
        self.load_dicom_btn = QPushButton("Load DICOM RT Dose")
        self.load_mcc_btn = QPushButton("Load MCC File")

        # 파일 로드 그룹
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        file_layout.addWidget(self.load_dicom_btn)
        file_layout.addWidget(self.load_mcc_btn)
        
        # 장비 정보 표시
        device_group = QGroupBox("Device Info")
        device_layout = QVBoxLayout(device_group)
        self.device_label = QLabel("Device Type: Not detected")
        self.origin_label = QLabel("Origin: Not set")
        device_layout.addWidget(self.device_label)
        device_layout.addWidget(self.origin_label)
        
        # 원점 조정 컨트롤
        origin_group = QGroupBox("Origin Adjustment")
        origin_layout = QGridLayout(origin_group)
        origin_layout.addWidget(QLabel("DICOM X (pixels):"), 0, 0)
        self.dicom_x_spin = QSpinBox()
        self.dicom_x_spin.setRange(-2000, 2000)
        self.dicom_x_spin.valueChanged.connect(self.update_origin)
        origin_layout.addWidget(self.dicom_x_spin, 0, 1)
        
        origin_layout.addWidget(QLabel("DICOM Y (pixels):"), 1, 0)
        self.dicom_y_spin = QSpinBox()
        self.dicom_y_spin.setRange(-2000, 2000)
        self.dicom_y_spin.valueChanged.connect(self.update_origin)
        origin_layout.addWidget(self.dicom_y_spin, 1, 1)
        
        # 프로파일 방향 선택 컨트롤
        profile_dir_group = QGroupBox("Profile Direction")
        profile_dir_layout = QVBoxLayout(profile_dir_group)
        self.vertical_btn = QPushButton("Vertical")
        self.horizontal_btn = QPushButton("Horizontal")
        self.vertical_btn.setCheckable(True)
        self.horizontal_btn.setCheckable(True)
        self.vertical_btn.setChecked(True)
        
        self.vertical_btn.clicked.connect(lambda: self.set_profile_direction("vertical"))
        self.horizontal_btn.clicked.connect(lambda: self.set_profile_direction("horizontal"))
        
        profile_dir_layout.addWidget(self.vertical_btn)
        profile_dir_layout.addWidget(self.horizontal_btn)
        
        # 감마 분석 매개변수
        gamma_group = QGroupBox("Gamma Analysis Parameters")
        gamma_layout = QGridLayout(gamma_group)
        gamma_layout.addWidget(QLabel("DTA (mm):"), 0, 0)
        self.dta_spin = QSpinBox()
        self.dta_spin.setRange(1, 10)
        self.dta_spin.setValue(3)
        gamma_layout.addWidget(self.dta_spin, 0, 1)
        
        gamma_layout.addWidget(QLabel("DD (%):"), 1, 0)
        self.dd_spin = QSpinBox()
        self.dd_spin.setRange(1, 10)
        self.dd_spin.setValue(3)
        gamma_layout.addWidget(self.dd_spin, 1, 1)
        
        gamma_layout.addWidget(QLabel("Analysis Type:"), 2, 0)
        self.gamma_type_combo = QComboBox()
        self.gamma_type_combo.addItems(["Global", "Local"])
        gamma_layout.addWidget(self.gamma_type_combo, 2, 1)
        
        # 실행 버튼
        self.run_gamma_btn = QPushButton("Run Gamma Analysis")
        self.generate_report_btn = QPushButton("Generate Report")
        self.generate_report_btn.setEnabled(False)

        run_report_group = QGroupBox("Execute")
        run_report_layout = QVBoxLayout(run_report_group)
        run_report_layout.addWidget(self.run_gamma_btn)
        run_report_layout.addWidget(self.generate_report_btn)
        
        control_layout.addWidget(file_group, 0, 1)
        control_layout.addWidget(device_group, 0, 2)
        control_layout.addWidget(origin_group, 0, 3)
        control_layout.addWidget(profile_dir_group, 0, 4)
        control_layout.addWidget(gamma_group, 0, 5)
        control_layout.addWidget(run_report_group, 0, 6)
        
        main_layout.addWidget(control_panel)
        
        viz_widget = QWidget()
        viz_layout = QGridLayout(viz_widget)
        
        plot_size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        fixed_width, fixed_height = 6, 5
        
        self.dicom_canvas = MatplotlibCanvas(self, width=fixed_width, height=fixed_height)
        self.dicom_canvas.setSizePolicy(plot_size_policy)
        self.dicom_label = QLabel("DICOM RT Dose: None")
        
        self.mcc_canvas = MatplotlibCanvas(self, width=fixed_width, height=fixed_height)
        self.mcc_canvas.setSizePolicy(plot_size_policy)
        self.mcc_label = QLabel("MCC File: None")
        
        self.profile_canvas = MatplotlibCanvas(self, width=fixed_width, height=fixed_height)
        self.profile_canvas.setSizePolicy(plot_size_policy)
        
        self.profile_table = ProfileDataTable()
        profile_scroll = QScrollArea()
        profile_scroll.setWidget(self.profile_table)
        profile_scroll.setWidgetResizable(True)
        
        profile_widget = QWidget()
        profile_layout = QHBoxLayout(profile_widget)
        profile_layout.addWidget(self.profile_canvas, 2)
        profile_layout.addWidget(profile_scroll, 1)
        
        self.gamma_canvas = MatplotlibCanvas(self, width=fixed_width, height=fixed_height)
        self.gamma_canvas.setSizePolicy(plot_size_policy)
        self.gamma_stats_label = QLabel("Gamma Statistics: Not calculated")
        
        dicom_widget = QWidget()
        QVBoxLayout(dicom_widget).addWidget(self.dicom_canvas)
        QVBoxLayout(dicom_widget).addWidget(self.dicom_label)
        
        mcc_widget = QWidget()
        QVBoxLayout(mcc_widget).addWidget(self.mcc_canvas)
        QVBoxLayout(mcc_widget).addWidget(self.mcc_label)
        
        gamma_widget = QWidget()
        QVBoxLayout(gamma_widget).addWidget(self.gamma_canvas)
        QVBoxLayout(gamma_widget).addWidget(self.gamma_stats_label)
        
        viz_layout.addWidget(dicom_widget, 0, 0)
        viz_layout.addWidget(profile_widget, 0, 1)
        viz_layout.addWidget(mcc_widget, 1, 0)
        viz_layout.addWidget(gamma_widget, 1, 1)
        
        main_layout.addWidget(viz_widget)
        
        self.load_dicom_btn.clicked.connect(self.load_dicom_file)
        self.load_mcc_btn.clicked.connect(self.load_mcc_file)
        self.run_gamma_btn.clicked.connect(self.run_gamma_analysis)
        self.generate_report_btn.clicked.connect(self.generate_report)
        self.dicom_canvas.mpl_connect('button_press_event', self.on_dicom_click)

    def _calculate_dose_bounds(self, data: StandardDoseData, threshold_percent=1, margin_mm=20):
        """Calculates physical bounds from a StandardDoseData object."""
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
        filename, _ = QFileDialog.getOpenFileName(self, "Open DICOM RT Dose File", "./", "DICOM Files (*.dcm);;All Files (*)", options=options)
        if not filename: return

        try:
            self.dicom_data = load_dcm(filename)
            self.dose_bounds = self._calculate_dose_bounds(self.dicom_data, threshold_percent=1, margin_mm=20)

            # Store initial state for origin adjustments
            self.initial_dicom_phys_coords = (self.dicom_data.x_coords.copy(), self.dicom_data.y_coords.copy())

            # Calculate the "pixel origin" for the UI, mirroring old behavior
            pos_x, _, pos_y = self.dicom_data.metadata['image_position_patient']
            spacing_x, spacing_y = self.dicom_data.metadata['pixel_spacing']

            # Note: This calculation is for UI consistency only.
            # The new architecture primarily uses physical coordinates.
            pixel_origin_x = int(round(pos_x / spacing_x)) + 1
            pixel_origin_y = int(round(pos_y / spacing_y)) - 1
            self.initial_dicom_pixel_origin = (pixel_origin_x, pixel_origin_y)

            self.dicom_x_spin.setValue(pixel_origin_x)
            self.dicom_y_spin.setValue(pixel_origin_y)

            self.redraw_all_images()

            self.dicom_label.setText(f"DICOM RT Dose: {os.path.basename(filename)}")
            self.origin_label.setText(f"DICOM Physical Origin: ({pos_x:.2f}, {pos_y:.2f}) mm, Spacing: {spacing_x:.2f} mm")

            if self.mcc_data is not None:
                self.set_default_profile_and_generate()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load DICOM file: {e}")
            logger.error(f"DICOM load error: {e}", exc_info=True)

    def load_mcc_file(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Open MCC File", "./", "MCC Files (*.mcc);;All Files (*)", options=options)
        if not filename: return
        
        try:
            self.mcc_data = load_mcc(filename)
            self.redraw_all_images()
            
            meta = self.mcc_data.metadata
            self.device_label.setText(f"Device Type: {meta['device']}")
            self.mcc_label.setText(f"MCC File: {os.path.basename(filename)}")

            if self.dicom_data is not None:
                self.set_default_profile_and_generate()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load MCC file: {e}")
            logger.error(f"MCC load error: {e}", exc_info=True)

    def set_default_profile_and_generate(self):
        if self.profile_direction == "vertical":
            self.profile_line = {"type": "vertical", "x": 0}
        else:
            self.profile_line = {"type": "horizontal", "y": 0}
        self.redraw_all_images()
        self.generate_profile()

    def redraw_all_images(self):
        if self.dicom_data:
            draw_image(
                canvas=self.dicom_canvas,
                image_data=self.dicom_data.data_grid,
                extent=self.dicom_data.physical_extent,
                title='DICOM RT Dose',
                colorbar_label='Dose (Gy)',
                show_origin=True, show_colorbar=True,
                apply_cropping=True, crop_bounds=self.dose_bounds,
                line=self.profile_line
            )
        if self.mcc_data:
            draw_image(
                canvas=self.mcc_canvas,
                image_data=self.mcc_data.data_grid,
                extent=self.mcc_data.physical_extent,
                title='MCC Data (Interpolated)',
                colorbar_label='Dose',
                show_origin=True, show_colorbar=True,
                apply_cropping=True, crop_bounds=self.dose_bounds,
                line=self.profile_line
            )

    def update_origin(self):
        if not self.dicom_data or not self.initial_dicom_phys_coords:
            return

        spacing_x, spacing_y = self.dicom_data.metadata['pixel_spacing']

        # Calculate physical offset from the change in pixel origin
        pixel_offset_x = self.dicom_x_spin.value() - self.initial_dicom_pixel_origin[0]
        pixel_offset_y = self.dicom_y_spin.value() - self.initial_dicom_pixel_origin[1]

        phys_offset_x = pixel_offset_x * spacing_x
        phys_offset_y = pixel_offset_y * spacing_y

        # Apply offset to the original coordinates
        self.dicom_data.x_coords = self.initial_dicom_phys_coords[0] + phys_offset_x
        self.dicom_data.y_coords = self.initial_dicom_phys_coords[1] + phys_offset_y

        self.redraw_all_images()
        self.generate_profile()

    def set_profile_direction(self, direction):
        self.profile_direction = direction
        self.vertical_btn.setChecked(direction == "vertical")
        self.horizontal_btn.setChecked(direction == "horizontal")
        if self.dicom_data:
            self.set_default_profile_and_generate()
    
    def on_dicom_click(self, event):
        if event.inaxes != self.dicom_canvas.axes or not self.dicom_data: return
        
        phys_x, phys_y = event.xdata, event.ydata
        self.profile_line = {"type": self.profile_direction, "x": phys_x} if self.profile_direction == "vertical" else {"type": "horizontal", "y": phys_y}
        self.redraw_all_images()
        self.generate_profile()

    def generate_profile(self):
        if self.profile_line is None or not self.dicom_data: return
        
        try:
            fixed_pos = self.profile_line["x"] if self.profile_line["type"] == "vertical" else self.profile_line["y"]
            self.current_profile_data = extract_profile_data(
                direction=self.profile_line["type"],
                fixed_position=fixed_pos,
                dicom_data=self.dicom_data,
                mcc_data=self.mcc_data
            )
            
            if not self.current_profile_data: return

            self.profile_canvas.fig.clear()
            self.profile_canvas.axes = self.profile_canvas.fig.add_subplot(111)
            
            phys_coords = self.current_profile_data['phys_coords']
            dicom_values = self.current_profile_data['dicom_values']
            
            self.profile_canvas.axes.plot(phys_coords, dicom_values, 'b-', label='RT dose')
            
            if 'mcc_interp' in self.current_profile_data:
                self.profile_canvas.axes.plot(phys_coords, self.current_profile_data['mcc_interp'], 'r-', label='Measurement (interpolated)')
                self.profile_canvas.axes.plot(self.current_profile_data['mcc_phys_coords'], self.current_profile_data['mcc_values'], 'ro', label='Measurement (original)', markersize=5)
            
            x_label = "Y Position (mm)" if self.profile_direction == "vertical" else "X Position (mm)"
            title_prefix = f"X={fixed_pos:.2f}mm" if self.profile_direction == "vertical" else f"Y={fixed_pos:.2f}mm"
            self.profile_canvas.axes.set_xlabel(x_label)
            self.profile_canvas.axes.set_ylabel('Dose (Gy)')
            self.profile_canvas.axes.set_title(f'Dose Profile: {title_prefix}')

            if self.dose_bounds:
                lims = (self.dose_bounds['min_y'], self.dose_bounds['max_y']) if self.profile_direction == "vertical" else (self.dose_bounds['min_x'], self.dose_bounds['max_x'])
                self.profile_canvas.axes.set_xlim(lims)

            self.profile_canvas.axes.legend()
            self.profile_canvas.axes.grid(True)
            self.profile_canvas.fig.tight_layout()
            self.profile_canvas.draw()
            
            if 'mcc_phys_coords' in self.current_profile_data:
                self.profile_table.update_data(self.current_profile_data['mcc_phys_coords'], self.current_profile_data['dicom_at_mcc'], self.current_profile_data['mcc_values'])
            else:
                self.profile_table.update_data(phys_coords, dicom_values)

        except Exception as e:
            logger.error(f"Profile generation error: {e}", exc_info=True)
            QMessageBox.warning(self, "Warning", f"Could not generate profile: {e}")

    def run_gamma_analysis(self):
        if not self.dicom_data or not self.mcc_data:
            QMessageBox.warning(self, "Warning", "Both DICOM and MCC data must be loaded.")
            return
            
        try:            
            dta, dd = self.dta_spin.value(), self.dd_spin.value()
            is_global = self.gamma_type_combo.currentText() == "Global"
            
            results = perform_gamma_analysis(self.mcc_data, self.dicom_data, dd, dta, is_global)

            self.gamma_map, self.gamma_stats, self.phys_extent, self.mcc_interp_data, self.dd_map, self.dta_map, self.dd_stats, self.dta_stats = results
            
            if 'pass_rate' in self.gamma_stats:
                draw_image(
                    canvas=self.gamma_canvas, image_data=self.gamma_map, extent=self.phys_extent,
                    title=f'Gamma Analysis: Pass Rate = {self.gamma_stats["pass_rate"]:.2f}%',
                    colorbar_label='Gamma Index', show_origin=True, show_colorbar=True,
                    apply_cropping=True, crop_bounds=self.dose_bounds
                )
                stats_text = f"Gamma Stats: Pass = {self.gamma_stats['pass_rate']:.2f}% | Mean = {self.gamma_stats['mean']:.3f} | Max = {self.gamma_stats['max']:.3f}"
                self.gamma_stats_label.setText(stats_text)
                self.generate_report_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Warning", "No valid gamma results.")
                self.generate_report_btn.setEnabled(False)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gamma analysis failed: {e}")
            logger.error(f"Gamma analysis error: {e}", exc_info=True)
            self.generate_report_btn.setEnabled(False)

    def generate_report(self):
        if self.gamma_stats is None:
            QMessageBox.warning(self, "Warning", "Run gamma analysis first.")
            return

        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            report_dir = os.path.join(base_dir, 'Report')
            os.makedirs(report_dir, exist_ok=True)
            
            patient_id = self.dicom_data.metadata.get('patient_id', 'Unknown')
            dicom_filename_base = os.path.splitext(os.path.basename(self.dicom_data.metadata.get('filename', 'file')))[0]
            default_path = os.path.join(report_dir, f"report_{patient_id}_{dicom_filename_base}.jpg")

            output_path, _ = QFileDialog.getSaveFileName(self, "Save Report", default_path, "JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf)")
            if not output_path: return

            ver_profile = extract_profile_data("vertical", 0, self.dicom_data, self.mcc_data)
            hor_profile = extract_profile_data("horizontal", 0, self.dicom_data, self.mcc_data)

            generate_report(
                output_path=output_path, dicom_data=self.dicom_data, mcc_data=self.mcc_data,
                gamma_map=self.gamma_map, gamma_stats=self.gamma_stats,
                dta=self.dta_spin.value(), dd=self.dd_spin.value(), suppression_level=10,
                ver_profile_data=ver_profile, hor_profile_data=hor_profile,
                mcc_interp_data=self.mcc_interp_data, dd_stats=self.dd_stats, dta_stats=self.dta_stats,
                dose_bounds=self.dose_bounds
            )
            QMessageBox.information(self, "Success", f"Report saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate report: {e}")
            logger.error(f"Report generation error: {e}", exc_info=True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Correcting the path for running as a script
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the pyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        application_path = sys._MEIPASS
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))

    # Add src to path to allow imports when running directly
    sys.path.insert(0, os.path.abspath(os.path.join(application_path, '..')))

    window = GammaAnalysisApp()
    window.show()
    sys.exit(app.exec_())
