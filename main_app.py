import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSpinBox, QDoubleSpinBox, QComboBox, QGridLayout, 
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
        self.dta_spin = QDoubleSpinBox()
        self.dta_spin.setRange(0.1, 10.0)
        self.dta_spin.setValue(3.0)
        self.dta_spin.setSingleStep(0.1)
        gamma_layout.addWidget(self.dta_spin, 0, 1)
        
        gamma_layout.addWidget(QLabel("DD (%):"), 1, 0)
        self.dd_spin = QDoubleSpinBox()
        self.dd_spin.setRange(0.1, 10.0)
        self.dd_spin.setValue(3.0)
        self.dd_spin.setSingleStep(0.1)
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
        
        # 픽셀 데이터 가져오기
        self.dicom_image = self.dicom_handler.get_pixel_data()
        self.dose_bounds = self.dicom_handler.dose_bounds
        
        # 원점 정보 가져오기
        self.dicom_origin_x, self.dicom_origin_y = self.dicom_handler.get_origin_coords()
        self.pixel_spacing, _ = self.dicom_handler.get_spacing()
        
        # 스핀박스 값 업데이트
        self.dicom_x_spin.setValue(self.dicom_origin_x)
        self.dicom_y_spin.setValue(self.dicom_origin_y)
        
        # DICOM 이미지 그리기
        draw_image(
            canvas=self.dicom_canvas,
            image_data=self.dicom_image,
            extent=self.dicom_handler.get_physical_extent(),
            title='DICOM RT Dose (10% Threshold)',
            colorbar_label='Dose (Gy)',
            show_origin=True,
            show_colorbar=True,
            apply_cropping=True,
            crop_bounds=self.dose_bounds
        )
        
        # 파일 레이블 업데이트
        self.dicom_filename = self.dicom_handler.get_filename()
        self.dicom_label.setText(f"DICOM RT Dose: {self.dicom_filename}")
        
        # 원점 정보 표시 업데이트
        self.origin_label.setText(f"DICOM 원점: ({self.dicom_origin_x}, {self.dicom_origin_y}) 픽셀, 간격: {self.pixel_spacing} mm")

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
        draw_image(
            canvas=self.mcc_canvas,
            image_data=self.mcc_handler.get_matrix_data(),
            extent=self.mcc_handler.get_physical_extent(),
            title='MCC Data (10% Threshold)',
            colorbar_label='Dose',
            show_origin=True,
            show_colorbar=True,
            apply_cropping=False
        )
        
        # 장비 정보 업데이트
        mcc_origin_x, mcc_origin_y = self.mcc_handler.get_origin_coords()
        mcc_spacing_x, mcc_spacing_y = self.mcc_handler.get_spacing()
        
        self.device_label.setText(f"Device Type: {self.mcc_handler.get_device_name()}")
        self.origin_label.setText(f"MCC Origin: ({mcc_origin_x}, {mcc_origin_y}) pixels, Spacing: {mcc_spacing_x}x{mcc_spacing_y} mm")
        
        # 파일 레이블 업데이트
        mcc_filename = self.mcc_handler.get_filename()
        self.mcc_label.setText(f"MCC File: {mcc_filename}")

    def update_origin_x(self):
        """DICOM 원점 x좌표 업데이트"""
        self.dicom_handler.update_origin(self.dicom_x_spin.value(), self.dicom_handler.dicom_origin_y)
        self.dicom_origin_x = self.dicom_handler.dicom_origin_x
        self.dose_bounds = self.dicom_handler.dose_bounds
        
        # 이미지 다시 그리기
        if self.dicom_handler.get_pixel_data() is not None:
            self.draw_direction_lines()
    
    def update_origin_y(self):
        """DICOM 원점 y좌표 업데이트"""
        self.dicom_handler.update_origin(self.dicom_handler.dicom_origin_x, self.dicom_y_spin.value())
        self.dicom_origin_y = self.dicom_handler.dicom_origin_y
        self.dose_bounds = self.dicom_handler.dose_bounds
        
        # 이미지 다시 그리기
        if self.dicom_handler.get_pixel_data() is not None:
            self.draw_direction_lines()

    def set_profile_direction(self, direction):
        """프로파일 방향 설정"""
        self.profile_direction = direction
        
        if direction == "vertical":
            self.vertical_btn.setChecked(True)
            self.horizontal_btn.setChecked(False)
        else:
            self.vertical_btn.setChecked(False)
            self.horizontal_btn.setChecked(True)
            
        # 현재 프로파일 라인 재설정
        self.profile_line = None
        
        # 방향선만 다시 그리기(스케일링 표시 제외)
        if self.dicom_image is not None:
            self.draw_direction_lines(show_colorbar=False)
    
    def draw_direction_lines(self, show_colorbar=False):
        """DICOM 및 MCC 이미지 모두에 방향선 그리기"""
        # 방향 표시선이 있는 DICOM 이미지 그리기
        if self.dicom_handler.get_pixel_data() is not None:
            line = None
            if self.profile_direction == "vertical":
                line = {"type": "vertical", "x": 0}
            else:
                line = {"type": "horizontal", "y": 0}
                
            draw_image(
                canvas=self.dicom_canvas,
                image_data=self.dicom_handler.get_pixel_data(),
                extent=self.dicom_handler.get_physical_extent(),
                title='DICOM RT Dose (10% Threshold)',
                colorbar_label='Dose (Gy)' if show_colorbar else None,
                show_origin=True,
                show_colorbar=show_colorbar,
                apply_cropping=True,
                crop_bounds=self.dose_bounds,
                line=line
            )
            
        # 방향 표시선이 있는 MCC 이미지 그리기
        if self.mcc_handler.get_matrix_data() is not None:
            line = None
            if self.profile_direction == "vertical":
                line = {"type": "vertical", "x": 0}
            else:
                line = {"type": "horizontal", "y": 0}
                
            draw_image(
                canvas=self.mcc_canvas,
                image_data=self.mcc_handler.get_matrix_data(),
                extent=self.mcc_handler.get_physical_extent(),
                title='MCC Data',
                colorbar_label='Dose' if show_colorbar else None,
                show_origin=True,
                show_colorbar=show_colorbar,
                apply_cropping=False,
                line=line
            )

    def on_dicom_click(self, event):
        """클릭 시 중심을 통과하는 수직/수평선 생성 - 물리적 좌표 사용"""
        if event.inaxes != self.dicom_canvas.axes or self.dicom_handler.get_pixel_data() is None:
            return
        
        try:
            # 클릭에서 물리적 좌표 가져오기(mm)
            phys_x, phys_y = event.xdata, event.ydata        
            
            # 물리적 좌표 저장
            if self.profile_direction == "vertical":
                # 수직선: x좌표 고정, y 변화
                self.profile_line = {"type": "vertical", "x": phys_x}
            else:
                # 수평선: y좌표 고정, x 변화
                self.profile_line = {"type": "horizontal", "y": phys_y}
            
            # 프로파일 라인이 있는 DICOM 이미지 그리기
            draw_image(
                canvas=self.dicom_canvas,
                image_data=self.dicom_handler.get_pixel_data(),
                extent=self.dicom_handler.get_physical_extent(),
                title='DICOM RT Dose',
                colorbar_label='Dose (Gy)',
                show_origin=True,
                show_colorbar=True,
                apply_cropping=True,
                crop_bounds=self.dose_bounds,
                line=self.profile_line
            )
            
            # 프로파일 라인이 있는 MCC 이미지 그리기(데이터가 있는 경우)
            if self.mcc_handler.get_matrix_data() is not None:
                draw_image(
                    canvas=self.mcc_canvas,
                    image_data=self.mcc_handler.get_matrix_data(),
                    extent=self.mcc_handler.get_physical_extent(),
                    title='MCC Data',
                    colorbar_label='Dose',
                    show_origin=True,
                    show_colorbar=True,
                    apply_cropping=False,
                    line=self.profile_line
                )
            
            # 정보 로깅
            if self.profile_direction == "vertical":
                logger.info(f"Vertical profile: Physical X={phys_x:.2f}mm")
            else:
                logger.info(f"Horizontal profile: Physical Y={phys_y:.2f}mm")
            
            # 프로파일 생성 및 표시
            self.generate_profile()
            
        except Exception as e:
            logger.error(f"프로파일 라인 생성 오류: {str(e)}")
            QMessageBox.warning(self, "Warning", f"프로파일 라인 생성 중 오류 발생: {str(e)}")

    def generate_profile(self):
        """선택된 라인을 따라 프로파일 생성 - 물리적 좌표 사용"""
        if self.profile_line is None or self.dicom_handler.get_pixel_data() is None:
            return
        
        try:
            # MCC 데이터 객체 준비
            mcc_data = None
            if self.mcc_handler.get_matrix_data() is not None:
                mcc_data = self.mcc_handler
                
            # 프로파일 데이터 추출
            self.current_profile_data = extract_profile_data(
                direction=self.profile_line["type"],
                fixed_position=self.profile_line["x"] if self.profile_line["type"] == "vertical" else self.profile_line["y"],
                dicom_handler=self.dicom_handler,
                mcc_handler=mcc_data
            )
            
            # 프로파일 그래프 생성
            self.profile_canvas.fig.clear()
            self.profile_canvas.axes = self.profile_canvas.fig.add_subplot(111)
            
            # 데이터 추출
            phys_coords = self.current_profile_data['phys_coords']
            dicom_values = self.current_profile_data['dicom_values']
            profile_type = self.current_profile_data['type']
            fixed_pos = self.current_profile_data['fixed_pos']
            
            # 방향에 따른 레이블 설정
            if profile_type == 'vertical':
                x_label = "Y Position (mm)"
                title_prefix = f"X={fixed_pos:.2f}mm"
            else:
                x_label = "X Position (mm)"
                title_prefix = f"Y={fixed_pos:.2f}mm"
            
            # DICOM 값 플롯
            self.profile_canvas.axes.plot(phys_coords, dicom_values, 'b-', label='RT dose')
            
            # MCC 값 플롯(있는 경우)
            if 'mcc_interp' in self.current_profile_data:
                mcc_interp = self.current_profile_data['mcc_interp']
                mcc_phys_coords = self.current_profile_data['mcc_phys_coords']
                mcc_values = self.current_profile_data['mcc_values']
                
                # 유효 값만 표시(NaN 아닌 값)
                valid_interp = ~np.isnan(mcc_interp)
                if np.any(valid_interp):
                    # 보간 데이터 표시
                    self.profile_canvas.axes.plot(
                        phys_coords[valid_interp], 
                        mcc_interp[valid_interp], 
                        'r-', label='Measurement (interpolated)')
                    
                    # 원본 데이터 포인트 표시
                    self.profile_canvas.axes.plot(
                        mcc_phys_coords, mcc_values, 
                        'ro', label='Measurement (original)', markersize=5)
            
            self.profile_canvas.axes.set_xlabel(x_label)
            self.profile_canvas.axes.set_ylabel('Dose (Gy)')
            self.profile_canvas.axes.set_title(f'Dose Profile: {title_prefix}')
            self.profile_canvas.axes.legend()
            self.profile_canvas.axes.grid(True)
            self.profile_canvas.fig.tight_layout()
            self.profile_canvas.draw()
            
            # 프로파일 테이블 업데이트
            if 'mcc_phys_coords' in self.current_profile_data and 'dicom_at_mcc' in self.current_profile_data:
                # MCC 위치와 해당 DICOM 값으로 테이블 생성
                self.profile_table.update_data(
                    self.current_profile_data['mcc_phys_coords'],
                    self.current_profile_data['dicom_at_mcc'],
                    self.current_profile_data['mcc_values']
                )
            else:
                # MCC 데이터 없음, DICOM 데이터만 표시
                self.profile_table.update_data(phys_coords, dicom_values)
                
        except Exception as e:
            logger.error(f"프로파일 생성 오류: {str(e)}")
            QMessageBox.warning(self, "Warning", f"프로파일 생성 중 오류 발생: {str(e)}")


    def run_gamma_analysis(self):
        """pymedphys.gamma를 이용한 감마 분석 수행(MCC를 참조 데이터로 사용)"""
        if self.dicom_handler.get_pixel_data() is None or self.mcc_handler.get_matrix_data() is None:
            QMessageBox.warning(
                self, "Warning", "두 데이터(DICOM과 MCC) 모두 로드해야 함")
            return
            
        try:            
            # 파라미터 설정
            distance_mm_threshold = self.dta_spin.value()  # mm
            dose_percent_threshold = self.dd_spin.value()  # %
            global_normalisation = self.gamma_type_combo.currentText() == "Global"
            
            # 감마 분석 수행(모듈화된 함수 사용)
            gamma_map, gamma_stats, phys_extent = perform_gamma_analysis(
                self.mcc_handler,  # 참조 데이터로 MCC 사용
                self.dicom_handler,  # 평가 데이터로 DICOM 사용
                distance_mm_threshold,
                dose_percent_threshold,
                global_normalisation
            )
            
            if 'pass_rate' in gamma_stats:
                # 결과 시각화
                self.gamma_canvas.fig.clear()
                self.gamma_canvas.axes = self.gamma_canvas.fig.add_subplot(111)
                
                # 마스킹된 감마 맵 생성
                masked_gamma = np.ma.masked_invalid(gamma_map)
                
                # 컬러맵 설정
                cmap = plt.cm.jet.copy()
                cmap.set_bad('white', 1.0)
                norm = Normalize(vmin=0, vmax=2)
                
                # 감마 맵 표시
                im = self.gamma_canvas.axes.imshow(
                    masked_gamma, 
                    cmap=cmap, 
                    norm=norm,
                    extent=phys_extent
                )
                self.gamma_canvas.fig.colorbar(im, ax=self.gamma_canvas.axes, label='Gamma Index')
                
                # 원점 표시
                self.gamma_canvas.axes.plot(0, 0, 'wo', markersize=3, markeredgecolor='black')
                
                # 축 레이블 설정
                self.gamma_canvas.axes.set_xlabel('X Position (mm)')
                self.gamma_canvas.axes.set_ylabel('Y Position (mm)')
                self.gamma_canvas.axes.set_title(f'Pass rate = {gamma_stats["pass_rate"]:.2f}%')
                
                self.gamma_canvas.fig.tight_layout()
                self.gamma_canvas.draw()
                
                # 통계 레이블 업데이트
                stats_text = (f"감마 통계: 통과율 = {gamma_stats['pass_rate']:.2f}% | "
                            f"평균 = {gamma_stats['mean']:.3f} | 최소 = {gamma_stats['min']:.3f} | "
                            f"최대 = {gamma_stats['max']:.3f}")
                self.gamma_stats_label.setText(stats_text)
                
                # 로그에 결과 기록
                logger.info(f"감마 분석 완료: {stats_text}")
            else:
                QMessageBox.warning(
                    self, "Warning", "유효한 감마 분석 결과가 없습니다.")
                
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"감마 분석 오류: {str(e)}")
            logger.error(f"감마 분석 오류: {str(e)}", exc_info=True)

    def generate_report(self):
        """분석 결과가 포함된 보고서 생성"""
        if self.dicom_handler.get_pixel_data() is None or self.mcc_handler.get_matrix_data() is None:
            QMessageBox.warning(
                self, "Warning", "두 데이터(DICOM과 MCC) 모두 로드해야 함")
            return
            
        try:
            # 사용자에게 보고서 파일 위치 요청
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Report", "./report.pdf", 
                "PDF Files (*.pdf);;All Files (*)", options=options)
                
            if not filename:
                return
                
            # 보고서용 새 그림 생성
            report_fig = plt.figure(figsize=(11.7, 8.3))  # A4 크기
            
            # 플롯용 그리드 설정
            gs = report_fig.add_gridspec(3, 2)
            
            # 제목 추가
            report_fig.suptitle(f"2D Gamma Analysis Report\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                               fontsize=16)
            
            # DICOM 및 MCC 이미지 추가
            ax1 = report_fig.add_subplot(gs[0, 0])
            ax1.imshow(self.dicom_handler.get_pixel_data(), cmap='jet')
            ax1.set_title(f'DICOM RT Dose: {self.dicom_handler.get_filename()}')
            
            ax2 = report_fig.add_subplot(gs[0, 1])
            ax2.imshow(self.mcc_handler.get_matrix_data(), cmap='jet')
            ax2.set_title(f'MCC Data: {self.mcc_handler.get_filename()}')
            
            # 프로파일 추가(사용 가능한 경우)
            if self.current_profile_data is not None:
                ax3 = report_fig.add_subplot(gs[1, :])
                ax3.plot(self.current_profile_data['phys_coords'], 
                        self.current_profile_data['dicom_values'], 
                        'b-', label='DICOM')
                
                if 'mcc_interp' in self.current_profile_data:
                    ax3.plot(self.current_profile_data['phys_coords'], 
                            self.current_profile_data['mcc_interp'], 
                            'r-', label='MCC (interpolated)')
                    
                ax3.set_xlabel('Position (mm)')
                ax3.set_ylabel('Dose')
                ax3.set_title('Dose Profile')
                ax3.legend()
                ax3.grid(True)
            
            # 감마 분석 추가(사용 가능한 경우)
            if hasattr(self, 'gamma_stats_label') and '통과율' in self.gamma_stats_label.text():
                # 캔버스에서 감마 데이터 추출
                masked_gamma = self.gamma_canvas.axes.images[0].get_array().data
                
                ax4 = report_fig.add_subplot(gs[2, :])
                
                # NaN/마스킹된 값에 대한 커스텀 컬러맵 생성(흰색)
                cmap = plt.cm.jet.copy()
                cmap.set_bad('white', 1.0)
                
                # 컬러바용 Normalize 객체 생성
                norm = Normalize(vmin=0, vmax=2)
                
                im = ax4.imshow(masked_gamma, cmap=cmap, norm=norm)
                plt.colorbar(im, ax=ax4, label='Gamma Index')
                
                # 레이블에서 통계 추가
                stats_text = self.gamma_stats_label.text().replace('감마 통계: ', '')
                ax4.set_title(f'Gamma Analysis: {stats_text}')
                
                # 매개변수 추가
                param_text = (f"Parameters: DTA = {self.dta_spin.value()} mm, "
                             f"DD = {self.dd_spin.value()}%, "
                             f"Type = {self.gamma_type_combo.currentText()}")
                ax4.text(0.5, -0.1, param_text, transform=ax4.transAxes, 
                        ha='center', fontsize=10)
            
            # 레이아웃 조정 및 저장
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(filename)
            plt.close(report_fig)
            
            logger.info(f"보고서 저장 완료: {filename}")
            QMessageBox.information(
                self, "Success", f"보고서 저장 완료: {filename}")
                
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"보고서 생성 오류: {str(e)}")
            logger.error(f"보고서 생성 오류: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GammaAnalysisApp()
    window.show()
    sys.exit(app.exec_())
