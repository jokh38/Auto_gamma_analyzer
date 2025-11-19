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
                             QScrollArea, QGroupBox, QMessageBox, QSplitter, QFrame)
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
        
        # Connect UI signals to controller methods
        self.connect_signals()

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
        
        # 1. File Selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        self.load_dicom_btn = QPushButton("Load File A (Left)")
        self.load_measurement_btn = QPushButton("Load File B (Right)")
        file_layout.addWidget(self.load_dicom_btn)
        file_layout.addWidget(self.load_measurement_btn)
        
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
        
        sidebar_layout.addWidget(file_group)
        sidebar_layout.addWidget(device_group)
        sidebar_layout.addWidget(origin_group)
        sidebar_layout.addWidget(profile_dir_group)
        sidebar_layout.addWidget(gamma_group)
        sidebar_layout.addWidget(run_report_group)
        sidebar_layout.addStretch() # Push everything up
        
        # --- Main Content Area (Right) ---
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        
        # Vertical Splitter: Top (Images) vs Bottom (Analysis)
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setHandleWidth(8)
        
        # Top Section: File A and File B Images
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setHandleWidth(8)
        
        # File A Container
        self.dicom_canvas = MatplotlibCanvas(self)
        self.dicom_label = QLabel("File A: None")
        self.dicom_label.setAlignment(Qt.AlignCenter)
        dicom_container = QWidget()
        dicom_layout = QVBoxLayout(dicom_container)
        dicom_layout.addWidget(self.dicom_canvas)
        dicom_layout.addWidget(self.dicom_label)
        
        # File B Container
        self.mcc_canvas = MatplotlibCanvas(self)
        self.mcc_label = QLabel("File B: None")
        self.mcc_label.setAlignment(Qt.AlignCenter)
        mcc_container = QWidget()
        mcc_layout = QVBoxLayout(mcc_container)
        mcc_layout.addWidget(self.mcc_canvas)
        mcc_layout.addWidget(self.mcc_label)
        
        top_splitter.addWidget(dicom_container)
        top_splitter.addWidget(mcc_container)
        top_layout.addWidget(top_splitter)
        
        # Bottom Section: Profile/Table and Gamma Map
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.setHandleWidth(8)
        
        # Profile & Table Section (Left side of bottom)
        profile_container = QWidget()
        profile_layout = QHBoxLayout(profile_container)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        
        # Splitter for Profile Plot vs Table
        profile_inner_splitter = QSplitter(Qt.Horizontal)
        
        self.profile_canvas = MatplotlibCanvas(self)
        self.profile_table = ProfileDataTable()
        profile_scroll = QScrollArea()
        profile_scroll.setWidget(self.profile_table)
        profile_scroll.setWidgetResizable(True)
        
        profile_inner_splitter.addWidget(self.profile_canvas)
        profile_inner_splitter.addWidget(profile_scroll)
        # Give more space to the table (ratio 1:1)
        profile_inner_splitter.setStretchFactor(0, 1) 
        profile_inner_splitter.setStretchFactor(1, 1)
        
        profile_layout.addWidget(profile_inner_splitter)
        
        # Gamma Map Section (Right side of bottom)
        gamma_container = QWidget()
        gamma_layout = QVBoxLayout(gamma_container)
        self.gamma_canvas = MatplotlibCanvas(self)
        self.gamma_stats_label = QLabel("Gamma Statistics: Not calculated")
        self.gamma_stats_label.setAlignment(Qt.AlignCenter)
        gamma_layout.addWidget(self.gamma_canvas)
        gamma_layout.addWidget(self.gamma_stats_label)
        
        bottom_splitter.addWidget(profile_container)
        bottom_splitter.addWidget(gamma_container)
        bottom_layout.addWidget(bottom_splitter)
        
        # Add Top and Bottom to Main Splitter
        main_splitter.addWidget(top_widget)
        main_splitter.addWidget(bottom_widget)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)
        
        content_layout.addWidget(main_splitter)
        
        # Add Sidebar and Content to Main Layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_widget)

    def connect_signals(self):
        """Connects all UI widget signals to the controller's methods."""
        self.load_dicom_btn.clicked.connect(self.controller.load_dicom_file)
        self.load_measurement_btn.clicked.connect(self.controller.load_measurement_file)
        self.run_gamma_btn.clicked.connect(self.controller.run_gamma_analysis)
        self.generate_report_btn.clicked.connect(self.controller.generate_report)
        
        self.dicom_x_spin.valueChanged.connect(self.controller.update_origin)
        self.dicom_y_spin.valueChanged.connect(self.controller.update_origin)

        self.vertical_btn.clicked.connect(lambda: self.controller.set_profile_direction("vertical"))
        self.horizontal_btn.clicked.connect(lambda: self.controller.set_profile_direction("horizontal"))

        self.dicom_canvas.mpl_connect('button_press_event', self.controller.on_dicom_click_handler)

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
