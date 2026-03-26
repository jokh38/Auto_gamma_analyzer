import sys
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSpinBox, QDoubleSpinBox, QComboBox, QGridLayout, 
                             QScrollArea, QGroupBox, QMessageBox, QSplitter, QFrame,
                             QSizePolicy)
from PyQt5.QtCore import Qt
from typing import Optional

from src.data_manager import DataManager
from src.ui_components import MatplotlibCanvas, ProfileDataTable, PlotManager
from src.app_controller import AppController
from src.styles import DARK_THEME_QSS
from src.app_controller import AppController

class GammaAnalysisApp(QMainWindow):
    """
    The main window (View) of the application.
    Responsibilities:
    - Initializing and laying out UI components.
    - Delegating all user actions to the AppController.
    - Holding references to UI widgets that the controller may need to update.
    """
    def __init__(self):
        super().__init__()
        
        # The view only holds the state of UI elements, not application data.
        self.profile_direction = "vertical"
        
        # Initialize managers and controller
        self.data_manager = DataManager()
        self.init_ui() # UI must be initialized before plot_manager
        self.plot_manager = PlotManager(
            self.data_manager, self.dicom_canvas, self.mcc_canvas,
            self.profile_canvas, self.gamma_canvas, self.profile_table
        )
        self.controller = AppController(self, self.data_manager, self.plot_manager)

        # Connect UI signals to controller methods
        self.connect_signals()

        # Apply Dark Theme
        self.setStyleSheet(DARK_THEME_QSS)

    def init_ui(self):
        """Initializes and lays out all UI components."""
        self.setWindowTitle('2D Gamma Analysis')
        self.setGeometry(100, 100, 1400, 900) # Increased default size
                
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget) # Main layout is Horizontal (Sidebar + Content)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- Left Sidebar ---
        sidebar = QFrame()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet("background-color: #252526; border-right: 1px solid #333;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        sidebar_layout.setSpacing(15)
        
        # 2. Device Info
        device_group = QGroupBox("Device Info")
        device_layout = QVBoxLayout(device_group)
        self.device_label = QLabel("Device Type: Not detected")
        self.origin_label = QLabel("Origin: Not set")
        device_layout.addWidget(self.device_label)
        device_layout.addWidget(self.origin_label)
        
        # 3. Origin Adjustment
        origin_group = QGroupBox("Origin Adjustment")
        origin_layout = QGridLayout(origin_group)
        self.dicom_x_spin = QSpinBox()
        self.dicom_x_spin.setRange(-2000, 2000)
        self.dicom_y_spin = QSpinBox()
        self.dicom_y_spin.setRange(-2000, 2000)
        origin_layout.addWidget(QLabel("DICOM X (px):"), 0, 0)
        origin_layout.addWidget(self.dicom_x_spin, 0, 1)
        origin_layout.addWidget(QLabel("DICOM Y (px):"), 1, 0)
        origin_layout.addWidget(self.dicom_y_spin, 1, 1)
        
        # 4. Profile Direction
        profile_dir_group = QGroupBox("Profile Direction")
        profile_dir_layout = QHBoxLayout(profile_dir_group)
        self.vertical_btn = QPushButton("Vertical")
        self.vertical_btn.setCheckable(True)
        self.vertical_btn.setChecked(True)
        self.horizontal_btn = QPushButton("Horizontal")
        self.horizontal_btn.setCheckable(True)
        profile_dir_layout.addWidget(self.vertical_btn)
        profile_dir_layout.addWidget(self.horizontal_btn)
        
        # 5. Gamma Parameters
        gamma_group = QGroupBox("Gamma Parameters")
        gamma_layout = QGridLayout(gamma_group)
        self.dta_spin = QSpinBox()
        self.dta_spin.setRange(1, 10)
        self.dta_spin.setValue(3)
        self.dd_spin = QSpinBox()
        self.dd_spin.setRange(1, 10)
        self.dd_spin.setValue(3)
        self.gamma_type_combo = QComboBox()
        self.gamma_type_combo.addItems(["Global", "Local"])
        gamma_layout.addWidget(QLabel("DTA (mm):"), 0, 0)
        gamma_layout.addWidget(self.dta_spin, 0, 1)
        gamma_layout.addWidget(QLabel("DD (%):"), 1, 0)
        gamma_layout.addWidget(self.dd_spin, 1, 1)
        gamma_layout.addWidget(QLabel("Type:"), 2, 0)
        gamma_layout.addWidget(self.gamma_type_combo, 2, 1)
        
        # 6. Execution
        run_report_group = QGroupBox("Actions")
        run_report_layout = QVBoxLayout(run_report_group)
        self.run_gamma_btn = QPushButton("Run Gamma Analysis")
        self.run_gamma_btn.setStyleSheet("background-color: #007acc; font-weight: bold;") # Highlight action button
        self.generate_report_btn = QPushButton("Generate Report")
        self.generate_report_btn.setEnabled(False)
        run_report_layout.addWidget(self.run_gamma_btn)
        run_report_layout.addWidget(self.generate_report_btn)
        
        sidebar_layout.addWidget(device_group)
        sidebar_layout.addWidget(origin_group)
        sidebar_layout.addWidget(profile_dir_group)
        sidebar_layout.addWidget(gamma_group)
        sidebar_layout.addWidget(run_report_group)
        
        sidebar_layout.addStretch() # Push everything up
        
        self.close_btn = QPushButton("Close App")
        self.close_btn.setObjectName("closeBtn")
        sidebar_layout.addWidget(self.close_btn)
        
        # --- Main Content Area (Right) ---
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        
        # --- Main Layout Strategy: 3 Columns ---
        # Col 1: File A (Top) / Profile (Bottom)
        # Col 2: File B (Top) / Gamma (Bottom)
        # Col 3: Profile Table (Spans full height)
        
        self.main_h_splitter = QSplitter(Qt.Horizontal)
        self.main_h_splitter.setHandleWidth(8)
        
        # --- Column 1: File A & Profile ---
        col1_splitter = QSplitter(Qt.Vertical)
        col1_splitter.setHandleWidth(8)
        
        # File A Container (Top)
        self.dicom_canvas = MatplotlibCanvas(self)
        dicom_header_layout = QHBoxLayout()
        self.load_dicom_btn = QPushButton("Load File A")
        self.load_dicom_btn.setObjectName("loadBtn")
        self.dicom_label = QLabel("File A: None")
        self.dicom_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.dicom_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        dicom_header_layout.addWidget(self.load_dicom_btn)
        dicom_header_layout.addWidget(self.dicom_label)
        dicom_header_layout.addStretch()
        
        dicom_container = QWidget()
        dicom_layout = QVBoxLayout(dicom_container)
        dicom_layout.addLayout(dicom_header_layout)
        dicom_layout.addWidget(self.dicom_canvas)
        
        # Profile Container (Bottom)
        profile_container = QWidget()
        profile_layout = QVBoxLayout(profile_container)
        
        profile_header_layout = QHBoxLayout()
        profile_title = QLabel("Profile Plot")
        profile_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        profile_title.setMinimumHeight(28) # rough equivalent of top button height
        profile_header_layout.addWidget(profile_title)
        profile_header_layout.addStretch()
        profile_header_layout.addWidget(QLabel("A Norm"))
        self.file_a_norm_spin = QDoubleSpinBox()
        self.file_a_norm_spin.setDecimals(3)
        self.file_a_norm_spin.setRange(0.001, 1000.0)
        self.file_a_norm_spin.setSingleStep(0.1)
        self.file_a_norm_spin.setValue(1.0)
        self.file_a_norm_spin.setFixedWidth(80)
        profile_header_layout.addWidget(self.file_a_norm_spin)
        
        self.profile_canvas = MatplotlibCanvas(self)
        profile_layout.addLayout(profile_header_layout)
        profile_layout.addWidget(self.profile_canvas)
        
        
        # We need to defer adding mcc_container or profile_container to col1/col2 splitters
        # until they are defined.
        
        # --- Column 2: File B & Gamma Map ---
        col2_splitter = QSplitter(Qt.Vertical)
        col2_splitter.setHandleWidth(8)
        
        # File B Container (Top)
        self.mcc_canvas = MatplotlibCanvas(self)
        mcc_header_layout = QHBoxLayout()
        self.load_measurement_btn = QPushButton("Load File B")
        self.load_measurement_btn.setObjectName("loadBtn")
        self.mcc_label = QLabel("File B: None")
        self.mcc_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.mcc_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        mcc_header_layout.addWidget(self.load_measurement_btn)
        mcc_header_layout.addWidget(self.mcc_label)
        mcc_header_layout.addStretch()
        
        mcc_container = QWidget()
        mcc_layout = QVBoxLayout(mcc_container)
        mcc_layout.addLayout(mcc_header_layout)
        mcc_layout.addWidget(self.mcc_canvas)
        
        # Gamma Map Container (Bottom)
        gamma_container = QWidget()
        gamma_layout = QVBoxLayout(gamma_container)
        
        gamma_header_layout = QHBoxLayout()
        gamma_title = QLabel("Gamma Analysis")
        gamma_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        gamma_title.setMinimumHeight(28) # rough equivalent of top button height
        gamma_header_layout.addWidget(gamma_title)
        gamma_header_layout.addStretch()
        gamma_header_layout.addWidget(QLabel("B Norm"))
        self.file_b_norm_spin = QDoubleSpinBox()
        self.file_b_norm_spin.setDecimals(3)
        self.file_b_norm_spin.setRange(0.001, 1000.0)
        self.file_b_norm_spin.setSingleStep(0.1)
        self.file_b_norm_spin.setValue(1.0)
        self.file_b_norm_spin.setFixedWidth(80)
        gamma_header_layout.addWidget(self.file_b_norm_spin)
        
        self.gamma_canvas = MatplotlibCanvas(self)
        gamma_layout.addLayout(gamma_header_layout)
        gamma_layout.addWidget(self.gamma_canvas)
        
        col1_splitter.addWidget(dicom_container)
        col1_splitter.addWidget(mcc_container)
        col1_splitter.setStretchFactor(0, 1)
        col1_splitter.setStretchFactor(1, 1)
        
        col2_splitter.addWidget(profile_container)
        col2_splitter.addWidget(gamma_container)
        col2_splitter.setStretchFactor(0, 1)
        col2_splitter.setStretchFactor(1, 1)
        
        # Synchronize vertical splitters for Col 1 and Col 2
        col1_splitter.splitterMoved.connect(
            lambda pos, index: col2_splitter.moveSplitter(pos, index))
        col2_splitter.splitterMoved.connect(
            lambda pos, index: col1_splitter.moveSplitter(pos, index))
            
        # --- Column 3: Profile Table ---
        self.profile_table = ProfileDataTable()
        profile_scroll = QScrollArea()
        profile_scroll.setWidget(self.profile_table)
        profile_scroll.setWidgetResizable(True)
        # Define a minimum width so it isn't completely squished
        profile_scroll.setMinimumWidth(250)
        
        # Add Columns to Main Horizontal Splitter
        self.main_h_splitter.addWidget(col1_splitter)
        self.main_h_splitter.addWidget(col2_splitter)
        self.main_h_splitter.addWidget(profile_scroll)
        
        # Ratios (e.g. 2:2:1 width allocation)
        self.main_h_splitter.setStretchFactor(0, 2)
        self.main_h_splitter.setStretchFactor(1, 2)
        self.main_h_splitter.setStretchFactor(2, 1)
        self.main_h_splitter.setChildrenCollapsible(False)
        # Force default widths to be perfectly equal for Col 1 and Col 2
        self.main_h_splitter.setSizes([1000, 1000, 500])
        
        content_layout.addWidget(self.main_h_splitter)
        
        # Add Sidebar and Content to Main Layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_widget)

    def connect_signals(self):
        """Connects all UI widget signals to the controller's methods."""
        self.load_dicom_btn.clicked.connect(self.controller.load_dicom_file)
        self.load_measurement_btn.clicked.connect(self.controller.load_measurement_file)
        self.run_gamma_btn.clicked.connect(self.controller.run_gamma_analysis)
        self.generate_report_btn.clicked.connect(self.controller.generate_report)
        self.close_btn.clicked.connect(self.close)
        
        self.dicom_x_spin.valueChanged.connect(self.controller.update_origin)
        self.dicom_y_spin.valueChanged.connect(self.controller.update_origin)

        self.vertical_btn.clicked.connect(lambda: self.controller.set_profile_direction("vertical"))
        self.horizontal_btn.clicked.connect(lambda: self.controller.set_profile_direction("horizontal"))
        self.file_a_norm_spin.valueChanged.connect(lambda value: self.controller.update_normalization("A", value))
        self.file_b_norm_spin.valueChanged.connect(lambda value: self.controller.update_normalization("B", value))

        self.dicom_canvas.mpl_connect('button_press_event', self.controller.on_dicom_click_handler)
        self.mcc_canvas.mpl_connect('button_press_event', self.controller.on_mcc_click_handler)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if getattr(sys, 'frozen', False):
        application_path = sys._MEIPASS
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(application_path, '..')))
    window = GammaAnalysisApp()
    window.show()
    sys.exit(app.exec_())
