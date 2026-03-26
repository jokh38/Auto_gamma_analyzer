import numpy as np
from scipy.interpolate import griddata, interpn
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, TwoSlopeNorm

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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        
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
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(['Position (mm)', 'A (cGy)', 'B (cGy)', 'Gamma'])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def set_profile_direction(self, direction):
        axis_label = "Y(mm)" if direction == "vertical" else "X(mm)"
        self.setHorizontalHeaderLabels([axis_label, 'A (cGy)', 'B (cGy)', 'Gamma'])
        
    def update_data(self, positions, dose_values, measurement_values=None, gamma_values=None):
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

                item_gamma = QTableWidgetItem("N/A")
                item_gamma.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(i, 3, item_gamma)
        else:
            valid_indices = ~np.isnan(measurement_values)
            valid_positions = positions[valid_indices]
            valid_dose_values = dose_values[valid_indices]
            valid_measurements = measurement_values[valid_indices]
            valid_gamma = None
            if gamma_values is not None:
                gamma_values = np.asarray(gamma_values)
                if len(gamma_values) == len(positions):
                    valid_gamma = gamma_values[valid_indices]

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

                gamma_text = "N/A"
                if valid_gamma is not None and i < len(valid_gamma) and np.isfinite(valid_gamma[i]):
                    gamma_text = f"{valid_gamma[i]:.3f}"

                item_gamma = QTableWidgetItem(gamma_text)
                item_gamma.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(i, 3, item_gamma)


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
        origin='upper',
        aspect='equal'
    )
    
    if show_colorbar and colorbar_label is not None:
        canvas.fig.colorbar(im, ax=canvas.axes, label=colorbar_label, use_gridspec=True, fraction=0.046, pad=0.04)
    
    if show_origin:
        canvas.axes.plot(0, 0, 'wo', markersize=3, markeredgecolor='black')
    
    if line is not None:
        if line["type"] == "vertical":
            canvas.axes.axvline(x=line["x"], color='white', linestyle='-', linewidth=2)
        else:
            canvas.axes.axhline(y=line["y"], color='white', linestyle='-', linewidth=2)
    
    canvas.axes.set_title(title)
    canvas.axes.set_aspect('equal', adjustable='box')
    
    canvas.fig.tight_layout()
    canvas.draw_idle()


from src.data_manager import DataManager
from src.analysis import extract_profile_data
from src.file_handlers import MCCFileHandler

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

    def _get_interpolated_file_b_data(self):
        """Interpolate File B onto File A's grid for display when File B is MCC."""
        dm = self.data_manager
        if not dm.file_a_handler or not dm.file_b_handler:
            return None, None

        if not isinstance(dm.file_b_handler, MCCFileHandler):
            return None, None

        if dm.mcc_interp_data is not None:
            return dm.mcc_interp_data, dm.file_a_handler.get_physical_extent()

        if not hasattr(dm.file_a_handler, 'phys_x_mesh') or not hasattr(dm.file_a_handler, 'phys_y_mesh'):
            return None, None

        mcc_data = dm.file_b_handler.get_pixel_data()
        if mcc_data is None:
            return None, None

        valid_mask = mcc_data >= 0
        if not np.any(valid_mask):
            return None, None

        interp_data = griddata(
            (dm.file_b_handler.phys_x_mesh[valid_mask], dm.file_b_handler.phys_y_mesh[valid_mask]),
            mcc_data[valid_mask],
            (dm.file_a_handler.phys_x_mesh, dm.file_a_handler.phys_y_mesh),
            method='linear',
            fill_value=0
        )
        return interp_data, dm.file_a_handler.get_physical_extent()
 
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
            if self.profile_table is not None:
                self.profile_table.set_profile_direction(profile_direction)
            self.profile_canvas.fig.clear()
            self.profile_canvas.axes = self.profile_canvas.fig.add_subplot(111)

            phys_coords = profile_data['phys_coords']
            dicom_values = profile_data['dicom_values']

            self.profile_canvas.axes.plot(phys_coords, dicom_values, 'b-', label='A')

            if 'mcc_values' in profile_data and 'mcc_phys_coords' in profile_data:
                self.profile_canvas.axes.plot(
                    profile_data['mcc_phys_coords'],
                    profile_data['mcc_values'],
                    'ro',
                    label='B',
                    markersize=5
                )

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
            self.profile_canvas.axes.grid(False)
            self.profile_canvas.fig.tight_layout()
            self.profile_canvas.draw()

            if 'mcc_phys_coords' in profile_data:
                gamma_values = self._get_profile_gamma_values(profile_data)
                self.profile_table.update_data(
                    profile_data['mcc_phys_coords'],
                    profile_data['dicom_at_mcc'],
                    profile_data['mcc_values'],
                    gamma_values=gamma_values
                )
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
            # Use interpolated MCC data to match the report view whenever possible.
            if dm.use_mcc_interpolation:
                file_b_data, file_b_extent = self._get_interpolated_file_b_data()
                title_suffix = ' (Interpolated)'
            else:
                file_b_data, file_b_extent = None, None
                title_suffix = ''

            if file_b_data is None or file_b_extent is None:
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
        """Draws the gamma map with pass values in blue/green and failing values in red."""
        dm = self.data_manager
        if dm.gamma_stats and 'pass_rate' in dm.gamma_stats:
            gamma_data = dm.gamma_map_interp if dm.gamma_map_interp is not None else dm.gamma_map
            self.gamma_canvas.fig.clear()
            self.gamma_canvas.axes = self.gamma_canvas.fig.add_subplot(111)

            gamma_display = np.array(gamma_data, dtype=float, copy=True)
            gamma_display[~np.isfinite(gamma_display)] = np.nan

            cmap = LinearSegmentedColormap.from_list(
                "gamma_pass_fail",
                [
                    (0.0, "#153b8a"),
                    (0.35, "#1fa3c8"),
                    (0.499, "#35b779"),
                    (0.5, "#fee08b"),
                    (0.72, "#f46d43"),
                    (1.0, "#b2182b"),
                ],
            )
            cmap.set_bad(color="#1e1e1e")

            finite_values = gamma_display[np.isfinite(gamma_display)]
            max_gamma = float(np.nanmax(finite_values)) if finite_values.size else 2.0
            norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(2.0, max_gamma))

            im = self.gamma_canvas.axes.imshow(
                gamma_display,
                cmap=cmap,
                norm=norm,
                extent=dm.phys_extent,
                origin='upper',
                aspect='equal'
            )
            self.gamma_canvas.fig.colorbar(
                im,
                ax=self.gamma_canvas.axes,
                label='Gamma Index',
                use_gridspec=True,
                fraction=0.046,
                pad=0.04
            )
            self.gamma_canvas.axes.plot(0, 0, 'wo', markersize=3, markeredgecolor='black')
            if dm.profile_line is not None:
                if dm.profile_line["type"] == "vertical":
                    self.gamma_canvas.axes.axvline(
                        x=dm.profile_line["x"], color='white', linestyle='-', linewidth=2
                    )
                else:
                    self.gamma_canvas.axes.axhline(
                        y=dm.profile_line["y"], color='white', linestyle='-', linewidth=2
                    )
            self.gamma_canvas.axes.set_aspect('equal', adjustable='box')
            self.gamma_canvas.axes.set_title(
                f'Pass Rate = {dm.gamma_stats["pass_rate"]:.2f}%'
            )
            self.gamma_canvas.fig.tight_layout()
            self.gamma_canvas.draw_idle()
        else:
            # Clear the canvas if there's no data
            self.gamma_canvas.fig.clear()
            self.gamma_canvas.axes = self.gamma_canvas.fig.add_subplot(111)
            self.gamma_canvas.axes.set_title("Gamma Analysis")
            self.gamma_canvas.draw_idle()

    def handle_image_click(self, event, profile_direction, source="A"):
        """
        Handles the logic for a click event on the File A or File B canvas.
        This will be called by the AppController.
        Snaps the clicked position to File B resolution if available.
        """
        dm = self.data_manager
        source_canvas = self.dicom_canvas if source == "A" else self.mcc_canvas
        source_handler = dm.file_a_handler if source == "A" else dm.file_b_handler
        legacy_data = dm.dicom_data if source == "A" else dm.mcc_data

        if event.inaxes != source_canvas.axes:
            return False
        if not source_handler and legacy_data is None:
            return False
        if event.xdata is None or event.ydata is None:
            return False

        phys_x, phys_y = event.xdata, event.ydata

        # Snap to File B resolution if available so the profile line stays aligned
        # with the measurement/reference grid used throughout the application.
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
        if dm.gamma_stats is not None:
            self.draw_gamma_map()
        profile_data = self.generate_profile_data()
        self.draw_profile(profile_data, profile_direction)
        return True

    def _get_profile_gamma_values(self, profile_data):
        """Map current profile sample positions to the available gamma result grid."""
        dm = self.data_manager
        gamma_map = dm.gamma_map
        profile_positions = profile_data.get('mcc_phys_coords')
        if gamma_map is None or profile_positions is None or len(profile_positions) == 0:
            return None

        reference_handler = dm.file_b_handler or dm.mcc_handler
        if reference_handler is None:
            return None

        try:
            if profile_data.get('type') == "vertical":
                fixed_position = profile_data.get('fixed_pos')
                fixed_axis_coords = reference_handler.phys_x_mesh[0, :]
                profile_axis_mesh = reference_handler.phys_y_mesh
                closest_idx = int(np.argmin(np.abs(fixed_axis_coords - fixed_position)))
                gamma_line = gamma_map[:, closest_idx]
                gamma_positions = profile_axis_mesh[:, closest_idx]
            else:
                fixed_position = profile_data.get('fixed_pos')
                fixed_axis_coords = reference_handler.phys_y_mesh[:, 0]
                profile_axis_mesh = reference_handler.phys_x_mesh
                closest_idx = int(np.argmin(np.abs(fixed_axis_coords - fixed_position)))
                gamma_line = gamma_map[closest_idx, :]
                gamma_positions = profile_axis_mesh[closest_idx, :]

            gamma_positions = np.asarray(gamma_positions, dtype=float)
            gamma_line = np.asarray(gamma_line, dtype=float)

            if gamma_positions.shape != gamma_line.shape:
                return None

            gamma_values = np.full(len(profile_positions), np.nan, dtype=float)
            for i, pos in enumerate(profile_positions):
                nearest_idx = int(np.argmin(np.abs(gamma_positions - pos)))
                if np.isclose(gamma_positions[nearest_idx], pos, atol=1e-6):
                    gamma_values[i] = gamma_line[nearest_idx]

            return gamma_values
        except Exception:
            return None
