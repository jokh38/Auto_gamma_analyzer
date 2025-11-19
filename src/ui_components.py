import numpy as np
from scipy.interpolate import interpn
from scipy.interpolate import interpn
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib 캔버스 위젯"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Use dark background style for plots
        plt.style.use('dark_background')
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Ensure figure background matches the UI background
        self.fig.patch.set_facecolor('#2b2b2b')
        
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Adjust layout to maximize space and ensure labels are visible
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
        
        # Update rcParams for better visibility
        self.update_plot_style()

    def update_plot_style(self):
        """Configure plot styles for dark theme"""
        params = {
            'axes.facecolor': '#1e1e1e',
            'axes.edgecolor': '#888888',
            'axes.labelcolor': '#e0e0e0',
            'xtick.color': '#e0e0e0',
            'ytick.color': '#e0e0e0',
            'text.color': '#e0e0e0',
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'grid.color': '#444444',
            'grid.linestyle': '--',
            'grid.alpha': 0.5
        }
        self.fig.set_facecolor('#2b2b2b')
        for key, val in params.items():
            plt.rcParams[key] = val


class ProfileDataTable(QTableWidget):
    """프로파일 데이터 표시용 테이블 위젯"""
    def __init__(self, parent=None):
        super(ProfileDataTable, self).__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(['Position (mm)', 'A (cGy)', 'B (cGy)'])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
    def update_data(self, positions, dose_values, measurement_values=None):
        if measurement_values is None or len(measurement_values) == 0:
            self.setRowCount(len(positions))
            for i, (pos, dose) in enumerate(zip(positions, dose_values)):
                item_pos = QTableWidgetItem(f"{pos:.1f}")
                item_pos.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(i, 0, item_pos)

                item_dose = QTableWidgetItem(f"{dose * 100:.1f}")  # Convert to cGy
                item_dose.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(i, 1, item_dose)

                item_na = QTableWidgetItem("N/A")
                item_na.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(i, 2, item_na)
        else:
            valid_indices = ~np.isnan(measurement_values)
            valid_positions = positions[valid_indices]
            valid_dose_values = dose_values[valid_indices]
            valid_measurements = measurement_values[valid_indices]

            self.setRowCount(len(valid_positions))

            for i, (pos, dose, meas) in enumerate(zip(valid_positions, valid_dose_values, valid_measurements)):
                item_pos = QTableWidgetItem(f"{pos:.1f}")
                item_pos.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(i, 0, item_pos)

                item_dose = QTableWidgetItem(f"{dose * 100:.1f}")  # Convert to cGy
                item_dose.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(i, 1, item_dose)

                item_meas = QTableWidgetItem(f"{meas * 100:.1f}")  # Convert to cGy
                item_meas.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(i, 2, item_meas)


def draw_image(canvas, image_data, extent, title, colorbar_label=None,
               show_origin=True, show_colorbar=True, line=None):
    """
    통합된 이미지 그리기 함수.
    이제 크롭된 ROI 데이터를 직접 받으므로, 확대/크롭 관련 로직이 제거되었습니다.
    """
    canvas.fig.clear()
    canvas.axes = canvas.fig.add_subplot(111)
    
    # Use origin='upper' for consistent coordinate system with reference code
    im = canvas.axes.imshow(
        image_data,
        cmap='jet',
        extent=extent,
        origin='upper'
    )
    
    if show_colorbar and colorbar_label is not None:
        canvas.fig.colorbar(im, ax=canvas.axes, label=colorbar_label)
    
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
        self.profile_y_max = None  # Store max dose for consistent Y-axis scaling
 
    def generate_profile_data(self) -> dict | None:
        """
        Extracts profile data based on the current state of the DataManager.
        Uses file_a_handler and file_b_handler for A/B file comparison.
        """
        dm = self.data_manager
        if dm.profile_line is None:
            return None

        # Use new A/B handler approach
        if dm.file_a_handler and dm.file_b_handler:
            fixed_pos = dm.profile_line.get("x") if dm.profile_line.get("type") == "vertical" else dm.profile_line.get("y")

            # Delegate profile data generation to the analysis function using handlers
            # file_a_handler is treated as "dicom" (evaluation), file_b_handler as "mcc" (reference)
            profile_data = extract_profile_data(
                direction=dm.profile_line["type"],
                fixed_position=fixed_pos,
                dicom_handler=dm.file_a_handler,
                mcc_handler=dm.file_b_handler
            )

            return profile_data

        # Legacy fallback using old handler references
        if dm.dicom_handler:
            fixed_pos = dm.profile_line.get("x") if dm.profile_line.get("type") == "vertical" else dm.profile_line.get("y")

            profile_data = extract_profile_data(
                direction=dm.profile_line["type"],
                fixed_position=fixed_pos,
                dicom_handler=dm.dicom_handler,
                mcc_handler=dm.mcc_handler
            )

            return profile_data

        return None

    def draw_profile(self, profile_data, profile_direction):
        """Draws the profile data on the profile canvas."""
        if not profile_data: return

        try:
            self.profile_canvas.fig.clear()
            self.profile_canvas.axes = self.profile_canvas.fig.add_subplot(111)

            phys_coords = profile_data['phys_coords']
            dicom_values = profile_data['dicom_values']

            self.profile_canvas.axes.plot(phys_coords, dicom_values, 'b-', label='A')

            if 'mcc_interp' in profile_data:
                self.profile_canvas.axes.plot(phys_coords, profile_data['mcc_interp'], 'r-', label='B (interpolated)')
                self.profile_canvas.axes.plot(profile_data['mcc_phys_coords'], profile_data['mcc_values'], 'ro', label='B (original)', markersize=5)

            fixed_pos = self.data_manager.profile_line["x"] if self.data_manager.profile_line["type"] == "vertical" else self.data_manager.profile_line["y"]
            x_label = "Y Position (mm)" if profile_direction == "vertical" else "X Position (mm)"
            title_prefix = f"X={fixed_pos:.2f}mm" if profile_direction == "vertical" else f"Y={fixed_pos:.2f}mm"

            self.profile_canvas.axes.set_xlabel(x_label)
            self.profile_canvas.axes.set_ylabel('Dose (Gy)')
            self.profile_canvas.axes.set_title(f'Dose Profile: {title_prefix}')

            if self.data_manager.dicom_roi:
                extent = self.data_manager.dicom_roi.physical_extent
                lims = (extent[2], extent[3]) if profile_direction == "vertical" else (extent[0], extent[1])
                self.profile_canvas.axes.set_xlim(lims)

            # Set fixed Y-axis limits based on max dose (110% of max)
            # Calculate and store max dose on first profile draw or when data changes
            if self.profile_y_max is None:
                # Get max dose from both datasets
                max_dose_a = np.max(dicom_values) if len(dicom_values) > 0 else 0
                max_dose_b = 0
                if 'mcc_interp' in profile_data:
                    mcc_interp_valid = profile_data['mcc_interp'][~np.isnan(profile_data['mcc_interp'])]
                    max_dose_b = np.max(mcc_interp_valid) if len(mcc_interp_valid) > 0 else 0
                elif 'mcc_values' in profile_data:
                    max_dose_b = np.max(profile_data['mcc_values']) if len(profile_data['mcc_values']) > 0 else 0

                # Use the overall max from the entire dataset if available
                if self.data_manager.file_a_handler:
                    data_a = self.data_manager.file_a_handler.get_pixel_data()
                    if data_a is not None:
                        max_dose_a = np.max(data_a)

                if self.data_manager.file_b_handler:
                    data_b = self.data_manager.file_b_handler.get_pixel_data()
                    if data_b is not None:
                        # Filter out invalid MCC data points
                        valid_b = data_b[data_b >= 0]
                        if len(valid_b) > 0:
                            max_dose_b = np.max(valid_b)

                self.profile_y_max = max(max_dose_a, max_dose_b)

            # Set Y-axis limits to 110% of max dose
            if self.profile_y_max > 0:
                self.profile_canvas.axes.set_ylim([0, self.profile_y_max * 1.1])

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
        """Redraws File A and File B images using handler data."""
        dm = self.data_manager

        # Reset profile Y-axis max when redrawing images (e.g., when new files are loaded)
        self.profile_y_max = None

        # Draw File A (top)
        if dm.file_a_handler:
            file_a_data = dm.file_a_handler.get_pixel_data()
            file_a_extent = dm.file_a_handler.get_physical_extent()
            if file_a_data is not None and file_a_extent is not None:
                draw_image(
                    canvas=self.dicom_canvas, image_data=file_a_data,
                    extent=file_a_extent, title='File A (Top)',
                    colorbar_label='Dose (Gy)', show_origin=True, show_colorbar=True,
                    line=dm.profile_line
                )
        elif dm.dicom_roi:
            # Legacy fallback
            draw_image(
                canvas=self.dicom_canvas, image_data=dm.dicom_roi.dose_grid,
                extent=dm.dicom_roi.physical_extent, title='File A (Top)',
                colorbar_label='Dose (Gy)', show_origin=True, show_colorbar=True,
                line=dm.profile_line
            )

        # Draw File B (bottom)
        if dm.file_b_handler:
            # Use mcc_interp_data if available (same resolution as DICOM/report)
            if dm.use_mcc_interpolation and dm.mcc_interp_data is not None:
                file_b_data = dm.mcc_interp_data
                # Use DICOM extent since mcc_interp_data is on DICOM grid
                file_b_extent = dm.file_a_handler.get_physical_extent() if dm.file_a_handler else dm.file_b_handler.get_physical_extent()
                title_suffix = ' (Interpolated)'
            else:
                file_b_data = dm.file_b_handler.get_pixel_data()
                file_b_extent = dm.file_b_handler.get_physical_extent()
                title_suffix = ''

            if file_b_data is not None and file_b_extent is not None:
                draw_image(
                    canvas=self.mcc_canvas, image_data=file_b_data,
                    extent=file_b_extent, title=f'File B (Bottom){title_suffix}',
                    colorbar_label='Dose (Gy)', show_origin=True, show_colorbar=True,
                    line=dm.profile_line
                )
        elif dm.mcc_roi:
            # Legacy fallback
            draw_image(
                canvas=self.mcc_canvas, image_data=dm.mcc_roi.dose_grid,
                extent=dm.mcc_roi.physical_extent, title='File B (Bottom)',
                colorbar_label='Dose', show_origin=True, show_colorbar=True,
                line=dm.profile_line
            )

    def draw_gamma_map(self):
        """Draws the gamma map on the gamma canvas using interpolated data for smooth visualization."""
        dm = self.data_manager
        if dm.gamma_stats and 'pass_rate' in dm.gamma_stats:
            # Use interpolated gamma map if available for smooth visualization
            gamma_data = dm.gamma_map_interp if dm.gamma_map_interp is not None else dm.gamma_map
            draw_image(
                canvas=self.gamma_canvas, image_data=gamma_data, extent=dm.phys_extent,
                title=f'Gamma Analysis: Pass Rate = {dm.gamma_stats["pass_rate"]:.2f}%',
                colorbar_label='Gamma Index', show_origin=True, show_colorbar=True
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
        Snaps the clicked position to File B resolution if available.
        """
        dm = self.data_manager
        # Check if File A is loaded (new approach) or dicom_data exists (legacy)
        if event.inaxes != self.dicom_canvas.axes:
            return False
        if not dm.file_a_handler and not dm.dicom_data:
            return False

        phys_x, phys_y = event.xdata, event.ydata

        # Snap to File B resolution if available (File B typically has lower resolution like MCC)
        # First try file_b_handler, then fall back to mcc_handler for backward compatibility
        spacing_handler = dm.file_b_handler if dm.file_b_handler else dm.mcc_handler

        if spacing_handler:
            spacing_x, spacing_y = spacing_handler.get_spacing()

            # Snap the clicked position to the nearest grid point
            if profile_direction == "vertical":
                # For vertical profile, snap x position
                phys_x = round(phys_x / spacing_x) * spacing_x
            else:
                # For horizontal profile, snap y position
                phys_y = round(phys_y / spacing_y) * spacing_y

        dm.profile_line = {"type": profile_direction, "x": phys_x} if profile_direction == "vertical" else {"type": "horizontal", "y": phys_y}

        self.redraw_all_images()
        profile_data = self.generate_profile_data()
        self.draw_profile(profile_data, profile_direction)
        return True
