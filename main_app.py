import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSpinBox, QComboBox, QGridLayout, 
                             QScrollArea, QGroupBox, QMessageBox)
from datetime import datetime

from utils import logger
from file_handlers import DicomFileHandler, MCCFileHandler
from ui_components import MatplotlibCanvas, ProfileDataTable, draw_image
from analysis import extract_profile_data, perform_gamma_analysis

class GammaAnalysisApp(QMainWindow):
    """2D 감마 분석 애플리케이션"""
    def __init__(self):
        super().__init__()
        
        # 변수 초기화
        self.dicom_handler = DicomFileHandler()
        self.mcc_handler = MCCFileHandler()
        self.dicom_image = None
        self.profile_line = None
        self.current_profile_data = None
        self.profile_direction = "vertical"  # 기본값: 수직
        self.dose_bounds = None # 크롭 경계
        
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
        origin_layout.addWidget(QLabel("DICOM X:"), 0, 0)
        self.dicom_x_spin = QSpinBox()
        self.dicom_x_spin.setRange(-1000, 1000)
        self.dicom_x_spin.valueChanged.connect(self.update_origin_x)
        origin_layout.addWidget(self.dicom_x_spin, 0, 1)
        
        origin_layout.addWidget(QLabel("DICOM Y:"), 1, 0)
        self.dicom_y_spin = QSpinBox()
        self.dicom_y_spin.setRange(-1000, 1000)
        self.dicom_y_spin.valueChanged.connect(self.update_origin_y)
        origin_layout.addWidget(self.dicom_y_spin, 1, 1)
        
        # 프로파일 방향 선택 컨트롤
        profile_dir_group = QGroupBox("Profile Direction")
        profile_dir_layout = QHBoxLayout(profile_dir_group)
        self.vertical_btn = QPushButton("Vertical")
        self.horizontal_btn = QPushButton("Horizontal")
        self.vertical_btn.setCheckable(True)
        self.horizontal_btn.setCheckable(True)
        self.vertical_btn.setChecked(True)  # 기본값: 수직
        
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
        self.dta_spin.setSingleStep(1)
        gamma_layout.addWidget(self.dta_spin, 0, 1)
        
        gamma_layout.addWidget(QLabel("DD (%):"), 1, 0)
        self.dd_spin = QSpinBox()
        self.dd_spin.setRange(1, 10)
        self.dd_spin.setValue(3)
        self.dd_spin.setSingleStep(1)
        gamma_layout.addWidget(self.dd_spin, 1, 1)
        
        # 감마 분석 유형 선택
        gamma_layout.addWidget(QLabel("Analysis Type:"), 2, 0)
        self.gamma_type_combo = QComboBox()
        self.gamma_type_combo.addItems(["Global", "Local"])
        gamma_layout.addWidget(self.gamma_type_combo, 2, 1)
        
        # 실행 버튼
        self.run_gamma_btn = QPushButton("Run Gamma Analysis")
        
        # 리포트 버튼
        self.generate_report_btn = QPushButton("Generate Report")
        
        # 컨트롤 추가
        control_layout.addWidget(self.load_dicom_btn, 0, 0)
        control_layout.addWidget(self.load_mcc_btn, 0, 1)
        control_layout.addWidget(device_group, 0, 2)
        control_layout.addWidget(origin_group, 0, 3)
        control_layout.addWidget(profile_dir_group, 0, 4)
        control_layout.addWidget(gamma_group, 0, 5)
        control_layout.addWidget(self.run_gamma_btn, 0, 6)
        control_layout.addWidget(self.generate_report_btn, 0, 7)
        
        # 메인 레이아웃에 컨트롤 패널 추가
        main_layout.addWidget(control_panel)
        
        # 시각화 영역 생성(4분할)
        viz_widget = QWidget()
        viz_layout = QGridLayout(viz_widget)
        
        # 모든 플롯 영역에 최소 크기 및 정책 설정
        plot_size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        
        # 그래프 크기 표준화
        fixed_width, fixed_height = 6, 5  # 인치
        
        # 4개의 플롯 영역 생성(균일한 크기)
        self.dicom_canvas = MatplotlibCanvas(self, width=fixed_width, height=fixed_height)
        self.dicom_canvas.setSizePolicy(plot_size_policy)
        self.dicom_label = QLabel("DICOM RT Dose: None")
        
        self.mcc_canvas = MatplotlibCanvas(self, width=fixed_width, height=fixed_height)
        self.mcc_canvas.setSizePolicy(plot_size_policy)
        self.mcc_label = QLabel("MCC File: None")
        
        self.profile_canvas = MatplotlibCanvas(self, width=fixed_width, height=fixed_height)
        self.profile_canvas.setSizePolicy(plot_size_policy)
        
        # 프로파일 테이블 생성 및 스크롤 영역에 추가
        self.profile_table = ProfileDataTable()
        profile_scroll = QScrollArea()
        profile_scroll.setWidget(self.profile_table)
        profile_scroll.setWidgetResizable(True)
        
        # 프로파일 그래프와 테이블이 있는 위젯 생성
        profile_widget = QWidget()
        profile_layout = QHBoxLayout(profile_widget)
        profile_layout.addWidget(self.profile_canvas, 2)
        profile_layout.addWidget(profile_scroll, 1)
        
        self.gamma_canvas = MatplotlibCanvas(self, width=fixed_width, height=fixed_height)
        self.gamma_canvas.setSizePolicy(plot_size_policy)
        self.gamma_stats_label = QLabel("Gamma Statistics: Not calculated")
        
        # 캔버스를 레이아웃에 추가
        dicom_widget = QWidget()
        dicom_layout = QVBoxLayout(dicom_widget)
        dicom_layout.addWidget(self.dicom_canvas)
        dicom_layout.addWidget(self.dicom_label)
        
        mcc_widget = QWidget()
        mcc_layout = QVBoxLayout(mcc_widget)
        mcc_layout.addWidget(self.mcc_canvas)
        mcc_layout.addWidget(self.mcc_label)
        
        gamma_widget = QWidget()
        gamma_layout = QVBoxLayout(gamma_widget)
        gamma_layout.addWidget(self.gamma_canvas)
        gamma_layout.addWidget(self.gamma_stats_label)
        
        # 위젯을 시각화 레이아웃에 추가
        viz_layout.addWidget(dicom_widget, 0, 0)
        viz_layout.addWidget(profile_widget, 0, 1)
        viz_layout.addWidget(mcc_widget, 1, 0)
        viz_layout.addWidget(gamma_widget, 1, 1)
        
        # 모든 그래프에 균일한 크기 적용
        viz_layout.setColumnStretch(0, 1)
        viz_layout.setColumnStretch(1, 1)
        viz_layout.setRowStretch(0, 1)
        viz_layout.setRowStretch(1, 1)
        
        # 모든 그래프에 대한 여백 및 간격 표준화
        viz_layout.setContentsMargins(10, 10, 10, 10)
        viz_layout.setSpacing(10)
        
        # 메인 레이아웃에 시각화 영역 추가
        main_layout.addWidget(viz_widget)
        
        # 시그널 연결
        self.load_dicom_btn.clicked.connect(self.load_dicom_file)
        self.load_mcc_btn.clicked.connect(self.load_mcc_file)
        self.run_gamma_btn.clicked.connect(self.run_gamma_analysis)
        self.generate_report_btn.clicked.connect(self.generate_report)
        
        # DICOM 이미지에서의 라인 선택 설정
        self.dicom_canvas.mpl_connect('button_press_event', self.on_dicom_click)
         
    def load_dicom_file(self):
        """DICOM RT dose 파일 로드"""
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open DICOM RT Dose File", "./", 
            "DICOM Files (*.dcm);;All Files (*)", options=options)
        
        if not filename:
            return
            
        result = self.dicom_handler.open_file(filename)
        
        if isinstance(result, tuple) and not result[0]:
            QMessageBox.critical(self, "Error", result[1])
            return
        elif result is False:
            QMessageBox.critical(self, "Error", "Failed to load DICOM file")
            return
        
        # 픽셀 데이터 및 크롭 경계 가져오기
        self.dicom_image = self.dicom_handler.get_pixel_data()
        self.dose_bounds = self.dicom_handler.dose_bounds
        
        # 원점 정보 가져오기
        self.dicom_origin_x, self.dicom_origin_y = self.dicom_handler.get_origin_coords()
        self.pixel_spacing, _ = self.dicom_handler.get_spacing()
        
        # 스핀박스 값 업데이트
        self.dicom_x_spin.setValue(self.dicom_origin_x)
        self.dicom_y_spin.setValue(self.dicom_origin_y)
        
        # DICOM 이미지 그리기
        self.redraw_all_images()
        
        # 파일 레이블 업데이트
        self.dicom_filename = self.dicom_handler.get_filename()
        self.dicom_label.setText(f"DICOM RT Dose: {self.dicom_filename}")
        
        # 원점 정보 표시 업데이트
        self.origin_label.setText(f"DICOM 원점: ({self.dicom_origin_x}, {self.dicom_origin_y}) 픽셀, 간격: {self.pixel_spacing} mm")

        # 두 파일이 모두 로드되었으면 기본 프로파일 생성
        if self.mcc_handler.get_matrix_data() is not None:
            self.set_default_profile_and_generate()

    def load_mcc_file(self):
        """MCC 파일 로드"""
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open MCC File", "./", 
            "MCC Files (*.mcc);;All Files (*)", options=options)
        
        if not filename:
            return
            
        result = self.mcc_handler.open_file(filename)
        
        if isinstance(result, tuple) and not result[0]:
            QMessageBox.critical(self, "Error", result[1])
            return
        elif result is False:
            QMessageBox.critical(self, "Error", "Failed to load MCC file")
            return
        
        # MCC 이미지 그리기
        self.redraw_all_images()
        
        # 장비 정보 업데이트
        mcc_origin_x, mcc_origin_y = self.mcc_handler.get_origin_coords()
        mcc_spacing_x, mcc_spacing_y = self.mcc_handler.get_spacing()
        
        self.device_label.setText(f"Device Type: {self.mcc_handler.get_device_name()}")
        self.origin_label.setText(f"MCC Origin: ({mcc_origin_x}, {mcc_origin_y}) pixels, Spacing: {mcc_spacing_x}x{mcc_spacing_y} mm")
        
        # 파일 레이블 업데이트
        mcc_filename = self.mcc_handler.get_filename()
        self.mcc_label.setText(f"MCC File: {mcc_filename}")

        # 두 파일이 모두 로드되었으면 기본 프로파일 생성
        if self.dicom_handler.get_pixel_data() is not None:
            self.set_default_profile_and_generate()

    def set_default_profile_and_generate(self):
        """(0,0)을 지나는 기본 프로파일을 설정하고 생성합니다."""
        if self.profile_direction == "vertical":
            self.profile_line = {"type": "vertical", "x": 0}
        else:
            self.profile_line = {"type": "horizontal", "y": 0}
        self.redraw_all_images()
        self.generate_profile()

    def redraw_all_images(self):
        """모든 이미지를 현재 상태에 맞게 다시 그립니다."""
        # DICOM 이미지 그리기
        if self.dicom_handler.get_pixel_data() is not None:
            draw_image(
                canvas=self.dicom_canvas,
                image_data=self.dicom_image,
                extent=self.dicom_handler.get_physical_extent(),
                title='DICOM RT Dose',
                colorbar_label='Dose (Gy)',
                show_origin=True,
                show_colorbar=True,
                apply_cropping=True,
                crop_bounds=self.dose_bounds,
                line=self.profile_line
            )

        # MCC 이미지 그리기 (보간된 데이터 사용)
        if self.mcc_handler.get_matrix_data() is not None:
            interpolated_mcc = self.mcc_handler.get_interpolated_matrix_data()
            draw_image(
                canvas=self.mcc_canvas,
                image_data=interpolated_mcc,
                extent=self.mcc_handler.get_physical_extent(),
                title='MCC Data (Interpolated)',
                colorbar_label='Dose',
                show_origin=True,
                show_colorbar=True,
                apply_cropping=True, # DICOM 크롭 영역에 맞춤
                crop_bounds=self.dose_bounds,
                line=self.profile_line
            )

    def update_origin_x(self):
        """DICOM 원점 x좌표 업데이트"""
        if self.dicom_handler.get_pixel_data() is None: return
        self.dicom_handler.update_origin(self.dicom_x_spin.value(), self.dicom_handler.dicom_origin_y)
        self.dicom_origin_x = self.dicom_handler.dicom_origin_x
        self.dose_bounds = self.dicom_handler.dose_bounds
        self.redraw_all_images()
    
    def update_origin_y(self):
        """DICOM 원점 y좌표 업데이트"""
        if self.dicom_handler.get_pixel_data() is None: return
        self.dicom_handler.update_origin(self.dicom_handler.dicom_origin_x, self.dicom_y_spin.value())
        self.dicom_origin_y = self.dicom_handler.dicom_origin_y
        self.dose_bounds = self.dicom_handler.dose_bounds
        self.redraw_all_images()

    def set_profile_direction(self, direction):
        """프로파일 방향 설정"""
        self.profile_direction = direction
        
        if direction == "vertical":
            self.vertical_btn.setChecked(True)
            self.horizontal_btn.setChecked(False)
        else:
            self.vertical_btn.setChecked(False)
            self.horizontal_btn.setChecked(True)
            
        self.profile_line = None
        if self.dicom_image is not None:
            self.redraw_all_images()
    
    def on_dicom_click(self, event):
        """클릭 시 중심을 통과하는 수직/수평선 생성 - 물리적 좌표 사용"""
        if event.inaxes != self.dicom_canvas.axes or self.dicom_handler.get_pixel_data() is None:
            return
        
        try:
            phys_x, phys_y = event.xdata, event.ydata        
            
            if self.profile_direction == "vertical":
                self.profile_line = {"type": "vertical", "x": phys_x}
            else:
                self.profile_line = {"type": "horizontal", "y": phys_y}
            
            self.redraw_all_images()
            
            if self.profile_direction == "vertical":
                logger.info(f"Vertical profile: Physical X={phys_x:.2f}mm")
            else:
                logger.info(f"Horizontal profile: Physical Y={phys_y:.2f}mm")
            
            self.generate_profile()
            
        except Exception as e:
            logger.error(f"프로파일 라인 생성 오류: {str(e)}")
            QMessageBox.warning(self, "Warning", f"프로파일 라인 생성 중 오류 발생: {str(e)}")

    def generate_profile(self):
        """선택된 라인을 따라 프로파일 생성 - 크롭 영역 반영"""
        if self.profile_line is None or self.dicom_handler.get_pixel_data() is None:
            return
        
        try:
            mcc_data = self.mcc_handler if self.mcc_handler.get_matrix_data() is not None else None
                
            self.current_profile_data = extract_profile_data(
                direction=self.profile_line["type"],
                fixed_position=self.profile_line["x"] if self.profile_line["type"] == "vertical" else self.profile_line["y"],
                dicom_handler=self.dicom_handler,
                mcc_handler=mcc_data
            )
            
            self.profile_canvas.fig.clear()
            self.profile_canvas.axes = self.profile_canvas.fig.add_subplot(111)
            
            phys_coords = self.current_profile_data['phys_coords']
            dicom_values = self.current_profile_data['dicom_values']
            profile_type = self.current_profile_data['type']
            fixed_pos = self.current_profile_data['fixed_pos']
            
            if profile_type == 'vertical':
                x_label, title_prefix = "Y Position (mm)", f"X={fixed_pos:.2f}mm"
            else:
                x_label, title_prefix = "X Position (mm)", f"Y={fixed_pos:.2f}mm"
            
            self.profile_canvas.axes.plot(phys_coords, dicom_values, 'b-', label='RT dose')
            
            if 'mcc_interp' in self.current_profile_data:
                mcc_interp = self.current_profile_data['mcc_interp']
                mcc_phys_coords = self.current_profile_data['mcc_phys_coords']
                mcc_values = self.current_profile_data['mcc_values']
                
                valid_interp = ~np.isnan(mcc_interp)
                if np.any(valid_interp):
                    self.profile_canvas.axes.plot(
                        phys_coords[valid_interp], mcc_interp[valid_interp], 
                        'r-', label='Measurement (interpolated)')
                    self.profile_canvas.axes.plot(
                        mcc_phys_coords, mcc_values, 
                        'ro', label='Measurement (original)', markersize=5)
            
            # 프로파일 뷰 크롭 적용
            if self.dose_bounds:
                if profile_type == 'vertical':
                    # For a vertical profile, the x-axis of the plot is the Y-position
                    self.profile_canvas.axes.set_xlim(self.dose_bounds['min_y'], self.dose_bounds['max_y'])
                else: # horizontal
                    # For a horizontal profile, the x-axis of the plot is the X-position
                    self.profile_canvas.axes.set_xlim(self.dose_bounds['min_x'], self.dose_bounds['max_x'])

            self.profile_canvas.axes.set_xlabel(x_label)
            self.profile_canvas.axes.set_ylabel('Dose (Gy)')
            self.profile_canvas.axes.set_title(f'Dose Profile: {title_prefix}')
            self.profile_canvas.axes.legend()
            self.profile_canvas.axes.grid(True)
            self.profile_canvas.fig.tight_layout()
            self.profile_canvas.draw()
            
            # 크롭된 데이터로 테이블 업데이트
            table_pos, table_dicom, table_mcc = phys_coords, dicom_values, None
            if self.dose_bounds:
                if profile_type == 'vertical':
                    mask = (phys_coords >= self.dose_bounds['min_y']) & (phys_coords <= self.dose_bounds['max_y'])
                else: # horizontal
                    mask = (phys_coords >= self.dose_bounds['min_x']) & (phys_coords <= self.dose_bounds['max_x'])
                
                table_pos = phys_coords[mask]
                table_dicom = dicom_values[mask]

            if 'mcc_phys_coords' in self.current_profile_data and 'dicom_at_mcc' in self.current_profile_data:
                mcc_pos = self.current_profile_data['mcc_phys_coords']
                dicom_at_mcc = self.current_profile_data['dicom_at_mcc']
                mcc_vals = self.current_profile_data['mcc_values']
                
                if self.dose_bounds:
                    if profile_type == 'vertical':
                        mcc_mask = (mcc_pos >= self.dose_bounds['min_y']) & (mcc_pos <= self.dose_bounds['max_y'])
                    else: # horizontal
                        mcc_mask = (mcc_pos >= self.dose_bounds['min_x']) & (mcc_pos <= self.dose_bounds['max_x'])
                    self.profile_table.update_data(mcc_pos[mcc_mask], dicom_at_mcc[mcc_mask], mcc_vals[mcc_mask])
                else:
                    self.profile_table.update_data(mcc_pos, dicom_at_mcc, mcc_vals)
            else:
                self.profile_table.update_data(table_pos, table_dicom)
                
        except Exception as e:
            logger.error(f"프로파일 생성 오류: {str(e)}")
            QMessageBox.warning(self, "Warning", f"프로파일 생성 중 오류 발생: {str(e)}")

    def run_gamma_analysis(self):
        """pymedphys.gamma를 이용한 감마 분석 수행(MCC를 참조 데이터로 사용)"""
        if self.dicom_handler.get_pixel_data() is None or self.mcc_handler.get_matrix_data() is None:
            QMessageBox.warning(self, "Warning", "두 데이터(DICOM과 MCC) 모두 로드해야 함")
            return
            
        try:            
            distance_mm_threshold = self.dta_spin.value()
            dose_percent_threshold = self.dd_spin.value()
            global_normalisation = self.gamma_type_combo.currentText() == "Global"
            
            gamma_map, gamma_stats, phys_extent = perform_gamma_analysis(
                self.mcc_handler, self.dicom_handler,
                dose_percent_threshold, distance_mm_threshold, global_normalisation
            )
            
            if 'pass_rate' in gamma_stats:
                draw_image(
                    canvas=self.gamma_canvas,
                    image_data=gamma_map,
                    extent=phys_extent,
                    title=f'Gamma Analysis: Pass Rate = {gamma_stats["pass_rate"]:.2f}%',
                    colorbar_label='Gamma Index',
                    show_origin=True,
                    show_colorbar=True,
                    apply_cropping=True, # 감마 맵 크롭
                    crop_bounds=self.dose_bounds
                )
                
                stats_text = (f"감마 통계: 통과율 = {gamma_stats['pass_rate']:.2f}% | "
                            f"평균 = {gamma_stats['mean']:.3f} | 최소 = {gamma_stats['min']:.3f} | "
                            f"최대 = {gamma_stats['max']:.3f}")
                self.gamma_stats_label.setText(stats_text)
                logger.info(f"감마 분석 완료: {stats_text}")
            else:
                QMessageBox.warning(self, "Warning", "유효한 감마 분석 결과가 없습니다.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"감마 분석 오류: {str(e)}")
            logger.error(f"감마 분석 오류: {str(e)}", exc_info=True)

    def generate_report(self):
        """분석 결과가 포함된 보고서 생성"""
        if self.dicom_handler.get_pixel_data() is None or self.mcc_handler.get_matrix_data() is None:
            QMessageBox.warning(self, "Warning", "두 데이터(DICOM과 MCC) 모두 로드해야 함")
            return
            
        try:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Report", "./report.pdf", 
                "PDF Files (*.pdf);;All Files (*)", options=options)
                
            if not filename:
                return
                
            report_fig = plt.figure(figsize=(11.7, 8.3))
            gs = report_fig.add_gridspec(3, 2)
            report_fig.suptitle(f"2D Gamma Analysis Report\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=16)
            
            # DICOM 이미지
            ax1 = report_fig.add_subplot(gs[0, 0])
            ax1.imshow(self.dicom_handler.get_pixel_data(), cmap='jet', extent=self.dicom_handler.get_physical_extent())
            if self.dose_bounds:
                ax1.set_xlim(self.dose_bounds['min_x'], self.dose_bounds['max_x'])
                ax1.set_ylim(self.dose_bounds['min_y'], self.dose_bounds['max_y'])
            ax1.set_title(f'DICOM RT Dose: {self.dicom_handler.get_filename()}')
            
            # MCC 이미지
            ax2 = report_fig.add_subplot(gs[0, 1])
            ax2.imshow(self.mcc_handler.get_interpolated_matrix_data(), cmap='jet', extent=self.mcc_handler.get_physical_extent())
            if self.dose_bounds:
                ax2.set_xlim(self.dose_bounds['min_x'], self.dose_bounds['max_x'])
                ax2.set_ylim(self.dose_bounds['min_y'], self.dose_bounds['max_y'])
            ax2.set_title(f'MCC Data: {self.mcc_handler.get_filename()}')
            
            # 프로파일
            if self.current_profile_data is not None:
                ax3 = report_fig.add_subplot(gs[1, :])
                ax3.plot(self.current_profile_data['phys_coords'], self.current_profile_data['dicom_values'], 'b-', label='DICOM')
                if 'mcc_interp' in self.current_profile_data:
                    ax3.plot(self.current_profile_data['phys_coords'], self.current_profile_data['mcc_interp'], 'r-', label='MCC (interpolated)')
                ax3.set_xlabel('Position (mm)')
                ax3.set_ylabel('Dose')
                ax3.set_title('Dose Profile')
                ax3.legend()
                ax3.grid(True)
            
            # 감마 분석
            if hasattr(self, 'gamma_stats_label') and '통과율' in self.gamma_stats_label.text():
                ax4 = report_fig.add_subplot(gs[2, :])
                gamma_image = self.gamma_canvas.axes.images[0]
                im = ax4.imshow(gamma_image.get_array(), cmap=gamma_image.get_cmap(), norm=gamma_image.get_norm(), extent=self.mcc_handler.get_physical_extent())
                if self.dose_bounds:
                    ax4.set_xlim(self.dose_bounds['min_x'], self.dose_bounds['max_x'])
                    ax4.set_ylim(self.dose_bounds['min_y'], self.dose_bounds['max_y'])
                plt.colorbar(im, ax=ax4, label='Gamma Index')
                stats_text = self.gamma_stats_label.text().replace('감마 통계: ', '')
                ax4.set_title(f'Gamma Analysis: {stats_text}')
                param_text = (f"Parameters: DTA = {self.dta_spin.value()} mm, DD = {self.dd_spin.value()}%, Type = {self.gamma_type_combo.currentText()}")
                ax4.text(0.5, -0.15, param_text, transform=ax4.transAxes, ha='center', fontsize=10)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(filename)
            plt.close(report_fig)
            
            logger.info(f"보고서 저장 완료: {filename}")
            QMessageBox.information(self, "Success", f"보고서 저장 완료: {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"보고서 생성 오류: {str(e)}")
            logger.error(f"보고서 생성 오류: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GammaAnalysisApp()
    window.show()
    sys.exit(app.exec_())
