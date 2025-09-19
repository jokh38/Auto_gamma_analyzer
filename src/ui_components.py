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
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)


class ProfileDataTable(QTableWidget):
    """프로파일 데이터 표시용 테이블 위젯"""
    def __init__(self, parent=None):
        super(ProfileDataTable, self).__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(['Position (mm)', 'RT dose (Gy)', 'Measurement'])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
    def update_data(self, positions, dose_values, measurement_values=None):
        if measurement_values is None or len(measurement_values) == 0:
            self.setRowCount(len(positions))
            for i, (pos, dose) in enumerate(zip(positions, dose_values)):
                self.setItem(i, 0, QTableWidgetItem(f"{pos:.1f}"))
                self.setItem(i, 1, QTableWidgetItem(f"{dose:.1f}"))
                self.setItem(i, 2, QTableWidgetItem("N/A"))
        else:
            valid_indices = ~np.isnan(measurement_values)
            valid_positions = positions[valid_indices]
            valid_dose_values = dose_values[valid_indices]
            valid_measurements = measurement_values[valid_indices]
            
            self.setRowCount(len(valid_positions))
            
            for i, (pos, dose, meas) in enumerate(zip(valid_positions, valid_dose_values, valid_measurements)):
                self.setItem(i, 0, QTableWidgetItem(f"{pos:.1f}"))
                self.setItem(i, 1, QTableWidgetItem(f"{dose:.1f}"))
                self.setItem(i, 2, QTableWidgetItem(f"{meas:.1f}"))


def draw_image(canvas, image_data, extent, title, colorbar_label=None,
               show_origin=True, show_colorbar=True, apply_cropping=False,
               crop_bounds=None, line=None):
    """
    통합된 이미지 그리기 함수.
    모든 데이터는 StandardDoseData 모델을 통해 좌표계가 통일되었으므로,
    조건부로 이미지를 뒤집는 로직을 제거했습니다.
    """
    canvas.fig.clear()
    canvas.axes = canvas.fig.add_subplot(111)
    
    # StandardDoseData 모델은 Y축이 항상 위로 향하므로 origin을 'lower'로 통일
    im = canvas.axes.imshow(
        image_data,
        cmap='jet',
        extent=extent,
        origin='lower'
    )
    
    if show_colorbar and colorbar_label is not None:
        canvas.fig.colorbar(im, ax=canvas.axes, label=colorbar_label)
    
    if apply_cropping and crop_bounds is not None:
        canvas.axes.set_xlim(crop_bounds['min_x'], crop_bounds['max_x'])
        canvas.axes.set_ylim(crop_bounds['min_y'], crop_bounds['max_y'])
    
    if show_origin:
        canvas.axes.plot(0, 0, 'wo', markersize=3, markeredgecolor='black')
    
    if line is not None:
        if line["type"] == "vertical":
            canvas.axes.axvline(x=line["x"], color='white', linestyle='-', linewidth=2)
        else:
            canvas.axes.axhline(y=line["y"], color='white', linestyle='-', linewidth=2)
    
    canvas.axes.set_title(title)
    
    canvas.fig.tight_layout()
    canvas.draw()
