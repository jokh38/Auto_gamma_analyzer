import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib 캔버스 위젯"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        초기화 함수
        
        Args:
            parent: 부모 위젯
            width: 그림 너비(인치)
            height: 그림 높이(인치)
            dpi: 해상도(dots per inch)
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        
        # 마진 최소화하여 캔버스 크기 최대화
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)


class ProfileDataTable(QTableWidget):
    """프로파일 데이터 표시용 테이블 위젯"""
    def __init__(self, parent=None):
        """
        초기화 함수
        
        Args:
            parent: 부모 위젯
        """
        super(ProfileDataTable, self).__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(['Position (mm)', 'RT dose (Gy)', 'Measurement'])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
    def update_data(self, positions, dose_values, measurement_values=None):
        """
        테이블 데이터 업데이트
        
        Args:
            positions: 위치 배열
            dose_values: 선량 값 배열
            measurement_values: 측정 값 배열(선택 사항)
        """
        if measurement_values is None or len(measurement_values) == 0:
            # 측정 값이 없으면 모든 포인트 표시
            self.setRowCount(len(positions))
            for i, (pos, dose) in enumerate(zip(positions, dose_values)):
                self.setItem(i, 0, QTableWidgetItem(f"{pos:.1f}"))
                self.setItem(i, 1, QTableWidgetItem(f"{dose:.1f}"))
                self.setItem(i, 2, QTableWidgetItem("N/A"))
        else:
            # 유효한 MCC 측정 값만 표시(NaN 아닌 값)
            valid_indices = ~np.isnan(measurement_values)
            valid_positions = positions[valid_indices]
            valid_dose_values = dose_values[valid_indices]
            valid_measurements = measurement_values[valid_indices]
            
            # 유효 데이터에 맞게 테이블 행 설정
            self.setRowCount(len(valid_positions))
            
            for i, (pos, dose, meas) in enumerate(zip(valid_positions, valid_dose_values, valid_measurements)):
                self.setItem(i, 0, QTableWidgetItem(f"{pos:.1f}"))
                self.setItem(i, 1, QTableWidgetItem(f"{dose:.1f}"))
                self.setItem(i, 2, QTableWidgetItem(f"{meas:.1f}"))


def draw_image(canvas, image_data, extent, title, colorbar_label=None,
               show_origin=True, show_colorbar=True, apply_cropping=False,
               crop_bounds=None, line=None, data_type='dicom'):
    """
    통합된 이미지 그리기 함수

    Args:
        canvas: 그림을 그릴 MatplotlibCanvas 객체
        image_data: 표시할 이미지 데이터 배열
        extent: [x_min, x_max, y_min, y_max] 범위
        title: 그래프 제목
        colorbar_label: 컬러바 라벨(선택 사항)
        show_origin: 원점 표시 여부(기본값: True)
        show_colorbar: 컬러바 표시 여부(기본값: True)
        apply_cropping: 크롭 적용 여부(기본값: False)
        crop_bounds: 도스 경계 정보 딕셔너리(선택 사항)
        line: 프로파일 라인 정보 딕셔너리(선택 사항, 예: {"type": "vertical", "x": value})
        data_type: 데이터 타입 ('dicom' 또는 'mcc'), 좌표계 설정용
    """
    # 캔버스 초기화
    canvas.fig.clear()
    canvas.axes = canvas.fig.add_subplot(111)
    
    image_to_display = image_data
    # DICOM, MCC, Gamma 맵 모두 Y축이 위로 향하도록 origin을 'lower'로 통일합니다.
    origin_to_use = 'lower'

    # 이미지 표시 - 데이터 타입에 따라 적절한 origin 설정
    # MCC와 Gamma 데이터는 배열의 위쪽(낮은 인덱스)에 물리적 Y좌표가 큰 값이 오므로,
    # 'lower' origin으로 올바르게 표시하려면 데이터를 상하 반전해야 합니다.
    if 'MCC' in title or 'Gamma' in title:
        image_to_display = np.flipud(image_data)

    im = canvas.axes.imshow(
        image_to_display,
        cmap='jet',
        extent=extent,
        origin=origin_to_use
    )
    
    # 컬러바 추가(옵션)
    if show_colorbar and colorbar_label is not None:
        canvas.fig.colorbar(im, ax=canvas.axes, label=colorbar_label)
    
    # 크롭 적용(옵션)
    if apply_cropping and crop_bounds is not None:
        canvas.axes.set_xlim(crop_bounds['min_x'], crop_bounds['max_x'])
        canvas.axes.set_ylim(crop_bounds['min_y'], crop_bounds['max_y'])
    
    # 원점 표시(옵션)
    if show_origin:
        canvas.axes.plot(0, 0, 'wo', markersize=3, markeredgecolor='black')
    
    # 프로파일 라인 표시(옵션)
    if line is not None:
        if line["type"] == "vertical":
            canvas.axes.axvline(x=line["x"], color='white', linestyle='-', linewidth=2)
        else:
            canvas.axes.axhline(y=line["y"], color='white', linestyle='-', linewidth=2)
    
    # 축 레이블 및 제목 설정
    canvas.axes.set_title(title)
    
    # 레이아웃 조정 및 그리기
    canvas.fig.tight_layout()
    canvas.draw()
