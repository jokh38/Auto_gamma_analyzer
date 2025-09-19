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
                             QScrollArea, QGroupBox, QMessageBox)
from typing import Optional

import sys
import os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QComboBox,
                             QGridLayout, QScrollArea, QGroupBox)

from src.data_manager import DataManager
from src.ui_components import MatplotlibCanvas, ProfileDataTable, PlotManager
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

    def init_ui(self):
        """Initializes and lays out all UI components."""
        self.setWindowTitle('2D Gamma Analysis')
        self.setGeometry(100, 100, 1000, 1000)
                
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        control_panel = QWidget()
        control_layout = QGridLayout(control_panel)
        
        self.load_dicom_btn = QPushButton("Load DICOM RT Dose")
        self.load_measurement_btn = QPushButton("Load Measurement File")

        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        file_layout.addWidget(self.load_dicom_btn)
        file_layout.addWidget(self.load_measurement_btn)
        
        self.device_label = QLabel("Device Type: Not detected")
        self.origin_label = QLabel("Origin: Not set")
        device_group = QGroupBox("Device Info")
        device_layout = QVBoxLayout(device_group)
        device_layout.addWidget(self.device_label)
        device_layout.addWidget(self.origin_label)
        
        self.dicom_x_spin = QSpinBox()
        self.dicom_x_spin.setRange(-2000, 2000)
        self.dicom_y_spin = QSpinBox()
        self.dicom_y_spin.setRange(-2000, 2000)
        origin_group = QGroupBox("Origin Adjustment")
        origin_layout = QGridLayout(origin_group)
        origin_layout.addWidget(QLabel("DICOM X (pixels):"), 0, 0)
        origin_layout.addWidget(self.dicom_x_spin, 0, 1)
        origin_layout.addWidget(QLabel("DICOM Y (pixels):"), 1, 0)
        origin_layout.addWidget(self.dicom_y_spin, 1, 1)
        
        self.vertical_btn = QPushButton("Vertical")
        self.vertical_btn.setCheckable(True)
        self.vertical_btn.setChecked(True)
        self.horizontal_btn = QPushButton("Horizontal")
        self.horizontal_btn.setCheckable(True)
        profile_dir_group = QGroupBox("Profile Direction")
        profile_dir_layout = QVBoxLayout(profile_dir_group)
        profile_dir_layout.addWidget(self.vertical_btn)
        profile_dir_layout.addWidget(self.horizontal_btn)
        
        self.dta_spin = QSpinBox()
        self.dta_spin.setRange(1, 10)
        self.dta_spin.setValue(3)
        self.dd_spin = QSpinBox()
        self.dd_spin.setRange(1, 10)
        self.dd_spin.setValue(3)
        self.gamma_type_combo = QComboBox()
        self.gamma_type_combo.addItems(["Global", "Local"])
        gamma_group = QGroupBox("Gamma Analysis Parameters")
        gamma_layout = QGridLayout(gamma_group)
        gamma_layout.addWidget(QLabel("DTA (mm):"), 0, 0)
        gamma_layout.addWidget(self.dta_spin, 0, 1)
        gamma_layout.addWidget(QLabel("DD (%):"), 1, 0)
        gamma_layout.addWidget(self.dd_spin, 1, 1)
        gamma_layout.addWidget(QLabel("Analysis Type:"), 2, 0)
        gamma_layout.addWidget(self.gamma_type_combo, 2, 1)
        
        self.run_gamma_btn = QPushButton("Run Gamma Analysis")
        self.generate_report_btn = QPushButton("Generate Report")
        self.generate_report_btn.setEnabled(False)
        run_report_group = QGroupBox("Execute")
        run_report_layout = QVBoxLayout(run_report_group)
        run_report_layout.addWidget(self.run_gamma_btn)
        run_report_layout.addWidget(self.generate_report_btn)
        
        control_layout.addWidget(file_group, 0, 1)
        control_layout.addWidget(device_group, 0, 2)
        control_layout.addWidget(origin_group, 0, 3)
        control_layout.addWidget(profile_dir_group, 0, 4)
        control_layout.addWidget(gamma_group, 0, 5)
        control_layout.addWidget(run_report_group, 0, 6)
        
        main_layout.addWidget(control_panel)
        
        viz_widget = QWidget()
        viz_layout = QGridLayout(viz_widget)
        
        self.dicom_canvas = MatplotlibCanvas(self)
        self.dicom_label = QLabel("DICOM RT Dose: None")
        dicom_widget = QWidget()
        QVBoxLayout(dicom_widget).addWidget(self.dicom_canvas)
        QVBoxLayout(dicom_widget).addWidget(self.dicom_label)
        
        self.mcc_canvas = MatplotlibCanvas(self)
        self.mcc_label = QLabel("Measurement File: None")
        mcc_widget = QWidget()
        QVBoxLayout(mcc_widget).addWidget(self.mcc_canvas)
        QVBoxLayout(mcc_widget).addWidget(self.mcc_label)
        
        self.profile_canvas = MatplotlibCanvas(self)
        self.profile_table = ProfileDataTable()
        profile_scroll = QScrollArea()
        profile_scroll.setWidget(self.profile_table)
        profile_scroll.setWidgetResizable(True)
        profile_widget = QWidget()
        profile_layout = QHBoxLayout(profile_widget)
        profile_layout.addWidget(self.profile_canvas, 2)
        profile_layout.addWidget(profile_scroll, 1)
        
        self.gamma_canvas = MatplotlibCanvas(self)
        self.gamma_stats_label = QLabel("Gamma Statistics: Not calculated")
        gamma_widget = QWidget()
        QVBoxLayout(gamma_widget).addWidget(self.gamma_canvas)
        QVBoxLayout(gamma_widget).addWidget(self.gamma_stats_label)
        
        viz_layout.addWidget(dicom_widget, 0, 0)
        viz_layout.addWidget(profile_widget, 0, 1)
        viz_layout.addWidget(mcc_widget, 1, 0)
        viz_layout.addWidget(gamma_widget, 1, 1)
        
        main_layout.addWidget(viz_widget)

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
