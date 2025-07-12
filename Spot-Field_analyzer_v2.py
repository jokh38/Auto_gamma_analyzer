import sys
import logging
import numpy as np
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
from pylinac.core.profile import FWXMProfilePhysical
from pylinac.core.profile import PenumbraLeftMetric, PenumbraRightMetric
from pylinac.core.profile import SymmetryPointDifferenceMetric, FlatnessDifferenceMetric
from scipy.optimize import curve_fit
import csv
from typing import List, Tuple

class ProfileAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_logging()
        self.mode = "field"
        self.matrix_octavius_mat = None
        self.device_task_type = None
        self.N_row = None
        self.init_ui()

    def setup_logging(self):
        """Configure logging for error tracking."""
        logging.basicConfig(
            filename="analyzer.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def init_ui(self):
        """Initialize the user interface with improved layout and functionality."""
        self.setWindowTitle("Advanced Profile & Spot Analyzer")
        self.resize(1300, 900)
        
        # Central widget and main layout
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # Top section with mode selection and file operations
        top_section = QtWidgets.QHBoxLayout()        
        self.btn_open = QtWidgets.QPushButton("Open .mcc File")
        self.btn_export = QtWidgets.QPushButton("Export Results")     
        top_section.addWidget(self.btn_open)
        top_section.addWidget(self.btn_export)
        
        sub_section = QtWidgets.QHBoxLayout()
        self.btn_field = QtWidgets.QPushButton("Field Analysis")
        self.btn_spot = QtWidgets.QPushButton("Spot Analysis")        
        sub_section.addWidget(self.btn_field)
        sub_section.addWidget(self.btn_spot)
        
        main_layout.addLayout(top_section)
        main_layout.addLayout(sub_section)
        
        # Main content area
        content_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # Left panel - Device selection and log
        left_panel = QtWidgets.QVBoxLayout()
        self.lst_device = QtWidgets.QListWidget()
        self.lst_device.addItems(["729", "729M", "1500", "1500M"])
        left_panel.addWidget(self.lst_device)
        
        self.log_text = QtWidgets.QTextBrowser()
        left_panel.addWidget(self.log_text)
                
        # Middle panel - Graphs 
        middle_panel = QtWidgets.QVBoxLayout()
        
        # Graphs
        self.spot_view = pg.PlotWidget()
        self.crline_view = pg.PlotWidget()
        self.inline_view = pg.PlotWidget()
        
        for view in [self.spot_view, self.crline_view, self.inline_view]:
            view.setBackground('w')
        
        graphs_layout = QtWidgets.QGridLayout()
        graphs_layout.addWidget(self.spot_view, 0, 0, 1, 2)
        self.spot_view.setTitle("Spot View")
        graphs_layout.addWidget(self.crline_view, 1, 0)
        self.crline_view.setTitle("Cross-line")
        graphs_layout.addWidget(self.inline_view, 1, 1)
        self.inline_view.setTitle("In-line")
        middle_panel.addLayout(graphs_layout)
                
        # Right panel - Results
        right_panel = QtWidgets.QVBoxLayout()
        
        # Results table
        self.result_table = QtWidgets.QTableWidget()
        right_panel.addWidget(self.result_table)
        
        # Add panels to content layout with stretch
        content_layout.addLayout(left_panel, 1)
        content_layout.addLayout(middle_panel, 6)
        content_layout.addLayout(right_panel, 4)
        
        
        # Connect signals
        self.btn_field.clicked.connect(self.field_analysis_with_results)
        self.btn_spot.clicked.connect(self.spot_analysis_with_results)
        self.btn_open.clicked.connect(self.open_file)
        self.btn_export.clicked.connect(self.export_results)
        
        # Default UI setup
        self.update_table_for_mode("field")
        
    def field_analysis_with_results(self):
        """Change mode to field and perform analysis automatically"""
        self.change_mode("field")
        if self.matrix_octavius_mat is not None:
            self.analyze_data()
    
    def spot_analysis_with_results(self):
        """Change mode to spot and perform analysis automatically"""
        self.change_mode("spot")
        if self.matrix_octavius_mat is not None:
            self.analyze_data()
            
    def change_mode(self, new_mode: str):
        """Switch between field and spot analysis modes."""
        self.mode = new_mode
        self.update_table_for_mode(new_mode)
        self.log_text.append(f"Switched to {new_mode.capitalize()} Analysis mode")
     
    def show_initial_view(self):
        """Display initial normalized matrix view."""
        if self.matrix_octavius_mat is None:
            return
        
        norm_array = self.matrix_octavius_mat.max()
        norm_data = self.matrix_octavius_mat / norm_array
        
        self.spot_view.clear()
        img = pg.ImageItem(norm_data)
        self.spot_view.addItem(img)
        
        self.crline_view.clear()
        self.inline_view.clear()
    
    def gaussian(self, x, y, x0, y0, xalpha, yalpha, A):
        """2D Gaussian function for spot analysis."""
        return A * np.exp(-((x-x0)/xalpha)**2 - ((y-y0)/yalpha)**2)
    
    def _gaussian(self, M, *args):
        """Wrapper for multi-gaussian fitting."""
        x, y = M
        arr = np.zeros(x.shape)
        for i in range(len(args)//5):
            arr += self.gaussian(x, y, *args[i*5:i*5+5])        
        return arr
    
    def analyze_field(self):
        """Perform field analysis on loaded data."""
        try:
            # Normalize data
            norm_data = self.matrix_octavius_mat / self.matrix_octavius_mat.max()
            
            # Crossline analysis
            profile_x = FWXMProfilePhysical(norm_data[26,:], fwxm_height=50, dpmm=0.2)
            profile_x_res = profile_x.as_resampled(interpolation_resolution_mm=1)
            
            x_results = {
                'center': (profile_x_res.center_idx-26)*0.5,
                'field_width': profile_x_res.field_width_px*0.5,
                'flatness': profile_x_res.compute([FlatnessDifferenceMetric(in_field_ratio=0.8)]),
                'symmetry': profile_x_res.compute([SymmetryPointDifferenceMetric(in_field_ratio=0.8)]),
                'left_penumbra': profile_x_res.compute([PenumbraLeftMetric()])*0.5,
                'right_penumbra': profile_x_res.compute([PenumbraRightMetric()])*0.5
            }
    
            # Inline analysis
            profile_y = FWXMProfilePhysical(norm_data[:,26], fwxm_height=50, dpmm=0.2)
            profile_y_res = profile_y.as_resampled(interpolation_resolution_mm=1)
            
            y_results = {
                'center': (profile_y_res.center_idx-26)*0.5,
                'field_width': profile_y_res.field_width_px*0.5,
                'flatness': profile_y_res.compute([FlatnessDifferenceMetric(in_field_ratio=0.8)]),
                'symmetry': profile_y_res.compute([SymmetryPointDifferenceMetric(in_field_ratio=0.8)]),
                'left_penumbra': profile_y_res.compute([PenumbraLeftMetric()])*0.5,
                'right_penumbra': profile_y_res.compute([PenumbraRightMetric()])*0.5
            }
    
            # Update table
            self.update_field_results(x_results, y_results)
            
            # Update graphs
            self.update_field_graphs(profile_x_res, profile_y_res)
    
        except Exception as e:
            logging.error(f"Field analysis error: {str(e)}")
            raise
    
    def analyze_spot(self):
        """Perform spot analysis on loaded data."""
        try:
            tmp_mat_array = self.matrix_octavius_mat
            xmin, xmax, nx = -13, 13, self.N_row
            ymin, ymax, ny = -13, 13, self.N_row
            x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
            X, Y = np.meshgrid(x, y)
    
            guess_prms = [
                (-5, 5, 1, 1, 1), (-5, 0, 1, 1, 1), (-5,-5, 1, 1, 1),
                (0, 5, 1, 1, 1), (0, 0, 1, 1, 1), (0,-5, 1, 1, 1),
                (5, 5, 1, 1, 1), (5, 0, 1, 1, 1), (5,-5, 1, 1, 1)
            ]
            
            p0 = [p for prms in guess_prms for p in prms]
            X_rav, Y_rav = X.ravel(), Y.ravel()
    
            if self.device_task_type == 2:
                xdata = np.vstack((X_rav[::2], Y_rav[::2]))
                Z_ravel = tmp_mat_array.ravel()[::2]
            else:            
                xdata = np.vstack((X_rav, Y_rav))
                Z_ravel = tmp_mat_array.ravel()
    
            popt, _ = curve_fit(self._gaussian, xdata, Z_ravel, p0)
            
            results = []
            for i in range(9):
                results.append({
                    'dx': popt[5*i] - guess_prms[i][0],
                    'dy': popt[5*i+1] - guess_prms[i][1],
                    'sx': popt[5*i+2],
                    'sy': popt[5*i+3]
                })
    
            self.update_spot_results(results)
            self.update_spot_graphs(popt, guess_prms)
    
        except Exception as e:
            logging.error(f"Spot analysis error: {str(e)}")
            raise
    
    def update_field_results(self, x_results, y_results):
        """Update field analysis results in the table."""
        try:
            result_keys = ['center', 'flatness', 'symmetry', 'left_penumbra', 'right_penumbra', 'field_width']
            for i, key in enumerate(result_keys):
                self.result_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{x_results[key]:.2f}"))
                self.result_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{y_results[key]:.2f}"))
        except Exception as e:
            logging.error(f"Field results update error: {str(e)}")
            raise
        self.result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)    
    
    def update_spot_results(self, results):
        """Update spot analysis results in the table."""
        try:
            for i, result in enumerate(results):
                self.result_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{result['dx']:.2f}"))
                self.result_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{result['dy']:.2f}"))
                self.result_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{result['sx']:.2f}"))
                self.result_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{result['sy']:.2f}"))
        except Exception as e:
            logging.error(f"Spot results update error: {str(e)}")
            raise
        self.result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)    
            
    def update_field_graphs(self, profile_x, profile_y):
        try:
            self.crline_view.clear()
            self.inline_view.clear()
            
            self.crline_view.plot(profile_x.x_values, profile_x.values)
            self.inline_view.plot(profile_y.x_values, profile_y.values)
            
            self.crline_view.setRange(xRange=[0, 50], yRange=[0, 1])
            self.inline_view.setRange(xRange=[0, 50], yRange=[0, 1])
        except Exception as e:
            logging.error(f"Field graphs update error: {str(e)}")
            raise
    
    def update_spot_graphs(self, popt, guess_prms):
        try:
            xmin, xmax = -13, 13
            nx = self.N_row * 10
            x = np.linspace(xmin, xmax, nx)
            X, Y = np.meshgrid(x, x)
            
            fit = np.zeros(X.shape)
            for i in range(len(popt)//5):
                fit += self.gaussian(X, Y, *popt[i*5:i*5+5])
            
            self.crline_view.clear()
            self.inline_view.clear()
            
            self.crline_view.plot(x, fit[:, nx//2])
            self.inline_view.plot(x, fit[nx//2, :])
            
            self.crline_view.plot(np.linspace(-13, 13, self.N_row), 
                                self.matrix_octavius_mat[:, self.N_row//2], 
                                pen=None, symbol='o')
            self.inline_view.plot(np.linspace(-13, 13, self.N_row), 
                                self.matrix_octavius_mat[self.N_row//2, :], 
                                pen=None, symbol='o')
            
            self.crline_view.setRange(xRange=[-25, 25], yRange=[0, 1])
            self.inline_view.setRange(xRange=[-25, 25], yRange=[0, 1])
        except Exception as e:
            logging.error(f"Spot graphs update error: {str(e)}")
            raise 
               
    def update_table_for_mode(self, mode):
        """Update table layout based on analysis mode."""
        self.result_table.clear()
        self.result_table.setColumnCount(2 if mode == "field" else 4)
        
        if mode == "field":
            self.result_table.setRowCount(6)
            self.result_table.setHorizontalHeaderLabels(['crossline','inline'])
            self.result_table.setVerticalHeaderLabels(['Center', 'Flatness', 'Symmetry', 'pen_L', 'pen_R','Field size'])
            self.result_table.horizontalHeader().setDefaultSectionSize(150)
            self.result_table.verticalHeader().setDefaultSectionSize(36)
        else:
            self.result_table.setRowCount(9)
            self.result_table.setHorizontalHeaderLabels(['Δx','Δy','σx','σy'])
            self.result_table.setVerticalHeaderLabels([f'Spot {i+1}' for i in range(9)])
            self.result_table.horizontalHeader().setDefaultSectionSize(76)
            self.result_table.verticalHeader().setDefaultSectionSize(25)
            
        # 내용에 맞게 열 너비 자동 조정
        # 수평 헤더 모드 설정
        self.result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    
    def open_file(self):
        """Enhanced file opening with robust error handling."""
        try:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open .mcc file", "./", "MCC files (*.mcc)"
            )
            if not filename:
                return
            
            self.log_text.clear()
            self.log_text.append(f"Reading: {filename}")
            
            with open(filename, "r") as file:
                lines = file.read().split()
            
            # Detect device and task type
            device_type, task_type = self.detect_device_type(lines)
            self.update_device_list(device_type, task_type)
            
            # Extract data
            N_begin = lines.count("BEGIN_DATA")
            self.N_row = N_begin
            self.matrix_octavius_mat = self._extract_data(lines, N_begin, device_type, task_type)
            
            # Initial visualization
            self.show_initial_view()
            self.log_text.append("File loaded successfully")
                        
        except Exception as e:
            error_msg = f"Error opening file: {str(e)}"
            self.log_text.append(error_msg)
            logging.error(error_msg)
            QtWidgets.QMessageBox.critical(self, "Error", error_msg)

    def _extract_data(self, lines, N_begin, device_type, task_type):
        """Extract measurement data from file lines."""
        try:
            delta_empty_detector = 1 if device_type == 2 and task_type == 1 else 0
            oct_read_intv = 2 if task_type == 2 else 3
            
            list_bged = np.zeros((N_begin, 3), dtype=int)
            ipp, ipp_b, ipp_f = -1, 0, 0
            
            for i, line in enumerate(lines):
                if line.startswith('BEGIN_DATA'):
                    list_bged[ipp_b, 0] = i
                    ipp_b += 1
                elif line.startswith('END_DATA'):
                    list_bged[ipp_f, 1] = i
                    list_bged[ipp_f, 2] = list_bged[ipp_f, 1] - list_bged[ipp_f, 0] - 1
                    ipp_f += 1
    
            matrix = np.zeros((N_begin, N_begin)) - 1
            x_lngt = int(list_bged[0, 2] / oct_read_intv)
    
            if device_type == 2 and task_type == 1:
                for j in range(0, N_begin, 2):
                    for k in range(0, x_lngt):
                        matrix[2*k, j] = float(lines[list_bged[j, 0] + oct_read_intv*k + 2])
                for j in range(1, N_begin, 2):
                    for k in range(0, x_lngt - delta_empty_detector):
                        matrix[2*k+1, j] = float(lines[list_bged[j, 0] + oct_read_intv*k + 2])
            else:
                for j in range(N_begin):
                    for k in range(x_lngt):
                        matrix[k, j] = float(lines[list_bged[j, 0] + oct_read_intv*k + 2])
    
            return matrix
        except Exception as e:
            logging.error(f"Data extraction error: {str(e)}")
            raise
        
    def detect_device_type(self, lines: List[str]) -> Tuple[int, int]:
        """Detect device and task type from file lines."""
        try:
            is_1500 = "SCAN_DEVICE=OCTAVIUS_1500_XDR" in lines
            is_merged = "SCAN_OFFAXIS_CROSSPLANE=0.00" in lines
            
            device_type = 2 if is_1500 else 1
            task_type = 2 if is_merged else 1
            
            return device_type, task_type
        except Exception as e:
            logging.error(f"Device type detection error: {str(e)}")
            raise
    
    def update_device_list(self, device_type: int, task_type: int):
        """Update device list highlighting based on detected type."""
        for i in range(4):
            self.lst_device.item(i).setBackground(QtGui.QColor("#FFFFFF"))
        
        device_task_type = (device_type - 1) * 2 + (task_type - 1)
        self.lst_device.item(device_task_type).setBackground(QtGui.QColor("#E3E3E3"))
        
        self.device_task_type = device_task_type
    
    def analyze_data(self):
        """Perform analysis based on current mode."""
        if self.matrix_octavius_mat is None or self.matrix_octavius_mat.size == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "Load a file first!")
            return
        
        try:
            if self.mode == "field":
                self.analyze_field()
            else:
                self.analyze_spot()
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            self.log_text.append(error_msg)
            logging.error(error_msg)
            QtWidgets.QMessageBox.critical(self, "Error", error_msg)
    
    def export_results(self):
        """Export results to CSV."""
        try:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export Results", "./", "CSV Files (*.csv)"
            )
            if not filename:
                return
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers
                headers = [self.result_table.horizontalHeaderItem(col).text() 
                           for col in range(self.result_table.columnCount())]
                writer.writerow(headers)
                
                # Write data
                for row in range(self.result_table.rowCount()):
                    row_data = [
                        self.result_table.item(row, col).text() 
                        if self.result_table.item(row, col) else ''
                        for col in range(self.result_table.columnCount())
                    ]
                    writer.writerow(row_data)
            
            self.log_text.append(f"Results exported to {filename}")
        except Exception as e:
            error_msg = f"Export error: {str(e)}"
            self.log_text.append(error_msg)
            logging.error(error_msg)
            QtWidgets.QMessageBox.critical(self, "Error", error_msg)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ProfileAnalyzer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()