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


from src.data_manager import DataManager
from src.analysis import extract_profile_data

class PlotManager:
    """
    Handles all plotting and visualization tasks.
    It is responsible for drawing on the canvases and preparing data for plots.
    """
    def __init__(self, data_manager: DataManager, dicom_canvas: MatplotlibCanvas,
                 mcc_canvas: MatplotlibCanvas, profile_canvas: MatplotlibCanvas,
                 gamma_canvas: MatplotlibCanvas, profile_table: ProfileDataTable):
        self.data_manager = data_manager
        self.dicom_canvas = dicom_canvas
        self.mcc_canvas = mcc_canvas
        self.profile_canvas = profile_canvas
        self.gamma_canvas = gamma_canvas
        self.profile_table = profile_table

    def generate_profile_data(self) -> dict | None:
        """
        Extracts profile data based on the current state of the DataManager.
        This method is designed to be testable without a GUI, as it returns data.
        """
        dm = self.data_manager
        if dm.profile_line is None or not dm.dicom_data:
            return None

        fixed_pos = dm.profile_line.get("x") if dm.profile_line.get("type") == "vertical" else dm.profile_line.get("y")

        profile_data = extract_profile_data(
            direction=dm.profile_line["type"],
            fixed_position=fixed_pos,
            dicom_data=dm.dicom_data,
            mcc_data=dm.mcc_data
        )

        return profile_data

    def draw_profile(self, profile_data, profile_direction):
        """Draws the profile data on the profile canvas."""
        if not profile_data: return

        try:
            self.profile_canvas.fig.clear()
            self.profile_canvas.axes = self.profile_canvas.fig.add_subplot(111)

            phys_coords = profile_data['phys_coords']
            dicom_values = profile_data['dicom_values']

            self.profile_canvas.axes.plot(phys_coords, dicom_values, 'b-', label='RT dose')

            if 'mcc_interp' in profile_data:
                self.profile_canvas.axes.plot(phys_coords, profile_data['mcc_interp'], 'r-', label='Measurement (interpolated)')
                self.profile_canvas.axes.plot(profile_data['mcc_phys_coords'], profile_data['mcc_values'], 'ro', label='Measurement (original)', markersize=5)

            fixed_pos = self.data_manager.profile_line["x"] if self.data_manager.profile_line["type"] == "vertical" else self.data_manager.profile_line["y"]
            x_label = "Y Position (mm)" if profile_direction == "vertical" else "X Position (mm)"
            title_prefix = f"X={fixed_pos:.2f}mm" if profile_direction == "vertical" else f"Y={fixed_pos:.2f}mm"

            self.profile_canvas.axes.set_xlabel(x_label)
            self.profile_canvas.axes.set_ylabel('Dose (Gy)')
            self.profile_canvas.axes.set_title(f'Dose Profile: {title_prefix}')

            if self.data_manager.dose_bounds:
                lims = (self.data_manager.dose_bounds['min_y'], self.data_manager.dose_bounds['max_y']) if profile_direction == "vertical" else (self.data_manager.dose_bounds['min_x'], self.data_manager.dose_bounds['max_x'])
                self.profile_canvas.axes.set_xlim(lims)

            self.profile_canvas.axes.legend()
            self.profile_canvas.axes.grid(True)
            self.profile_canvas.fig.tight_layout()
            self.profile_canvas.draw()

            if 'mcc_phys_coords' in profile_data:
                self.profile_table.update_data(profile_data['mcc_phys_coords'], profile_data['dicom_at_mcc'], profile_data['mcc_values'])
            else:
                self.profile_table.update_data(phys_coords, dicom_values)

        except Exception as e:
            # It's better to log errors than to show a message box from here,
            # as this class shouldn't be aware of high-level UI elements like QMessageBox.
            # The controller can decide to show a message.
            print(f"Error drawing profile: {e}") # Using print for now, logger would be better

    def redraw_all_images(self):
        """Redraws the DICOM and MCC images."""
        dm = self.data_manager
        if dm.dicom_data:
            draw_image(
                canvas=self.dicom_canvas, image_data=dm.dicom_data.data_grid,
                extent=dm.dicom_data.physical_extent, title='DICOM RT Dose',
                colorbar_label='Dose (Gy)', show_origin=True, show_colorbar=True,
                apply_cropping=True, crop_bounds=dm.dose_bounds, line=dm.profile_line
            )
        if dm.mcc_data:
            draw_image(
                canvas=self.mcc_canvas, image_data=dm.mcc_data.data_grid,
                extent=dm.mcc_data.physical_extent, title='MCC Data (Interpolated)',
                colorbar_label='Dose', show_origin=True, show_colorbar=True,
                apply_cropping=True, crop_bounds=dm.dose_bounds, line=dm.profile_line
            )

    def draw_gamma_map(self):
        """Draws the gamma map on the gamma canvas."""
        dm = self.data_manager
        if dm.gamma_stats and 'pass_rate' in dm.gamma_stats:
            draw_image(
                canvas=self.gamma_canvas, image_data=dm.gamma_map, extent=dm.phys_extent,
                title=f'Gamma Analysis: Pass Rate = {dm.gamma_stats["pass_rate"]:.2f}%',
                colorbar_label='Gamma Index', show_origin=True, show_colorbar=True,
                apply_cropping=True, crop_bounds=dm.dose_bounds
            )
        else:
            # Clear the canvas if there's no data
            self.gamma_canvas.fig.clear()
            self.gamma_canvas.axes = self.gamma_canvas.fig.add_subplot(111)
            self.gamma_canvas.axes.set_title("Gamma Analysis")
            self.gamma_canvas.draw()

    def handle_dicom_click(self, event, profile_direction):
        """
        Handles the logic for a click event on the DICOM canvas.
        This will be called by the AppController.
        """
        dm = self.data_manager
        if event.inaxes != self.dicom_canvas.axes or not dm.dicom_data:
            return False

        phys_x, phys_y = event.xdata, event.ydata
        dm.profile_line = {"type": profile_direction, "x": phys_x} if profile_direction == "vertical" else {"type": "horizontal", "y": phys_y}

        self.redraw_all_images()
        profile_data = self.generate_profile_data()
        self.draw_profile(profile_data, profile_direction)
        return True
