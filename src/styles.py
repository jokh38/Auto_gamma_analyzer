"""
Modern Dark Theme Styles for the Auto Gamma Analyzer
"""

DARK_THEME_QSS = """
/* Global Styles */
QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 10pt;
}

QMainWindow {
    background-color: #2b2b2b;
}

/* GroupBox */
QGroupBox {
    border: 1px solid #4d4d4d;
    border-radius: 6px;
    margin-top: 1.2em;
    padding-top: 10px;
    font-weight: bold;
    color: #4fa3d1; /* Accent color for titles */
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
}

/* Buttons */
QPushButton {
    background-color: #3c3f41;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 12px;
    color: #e0e0e0;
    min-height: 24px;
}

QPushButton:hover {
    background-color: #4e5254;
    border-color: #4fa3d1;
}

QPushButton:pressed {
    background-color: #2d2f31;
}

QPushButton:checked {
    background-color: #4fa3d1;
    color: #ffffff; /* White text on blue background */
    border-color: #4fa3d1;
    font-weight: bold;
}

QPushButton:disabled {
    background-color: #2b2b2b;
    color: #666;
    border-color: #333;
}

/* Specific Buttons */
QPushButton#loadBtn {
    background-color: #2a3d4a;
    border: 1px solid #4fa3d1;
    color: #81c9ff;
    font-weight: bold;
    min-height: 20px;
}

QPushButton#loadBtn:hover {
    background-color: #354a5a;
    border-color: #6ebbf5;
}

QPushButton#loadBtn:pressed {
    background-color: #1f2c36;
}

QPushButton#rotateBtn {
    font-size: 18px;
    padding: 0px;
    min-height: 0px;
}

QPushButton#closeBtn {
    background-color: #3c2a2a;
    border: 1px solid #a34f4f;
    color: #ff8181;
    font-weight: bold;
    padding: 8px 12px;
}

QPushButton#closeBtn:hover {
    background-color: #5a3535;
    border-color: #f56e6e;
    color: #ffffff;
}

QPushButton#closeBtn:pressed {
    background-color: #2c1f1f;
}

QPushButton#tabBtn {
    background-color: #1e2227;
    border: 1px solid #2c313a;
    color: #e0e0e0;
}

QPushButton#tabBtn:hover {
    background-color: #2a3036;
    border-color: #4fa3d1;
}

QPushButton#tabBtn:checked {
    background-color: #4fa3d1;
    color: #ffffff;
    font-weight: bold;
}

/* Inputs */
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1e1e1e;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 4px;
    color: #e0e0e0;
    selection-background-color: #4fa3d1;
    selection-color: #1e1e1e;
}

QSpinBox:focus, QComboBox:focus {
    border-color: #4fa3d1;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 0px;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
}

/* Labels */
QLabel {
    color: #cccccc;
}

/* ScrollBars */
QScrollBar:vertical {
    border: none;
    background: #2b2b2b;
    width: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #555;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #2b2b2b;
    height: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:horizontal {
    background: #555;
    min-width: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Splitter */
QSplitter::handle {
    background-color: #3c3f41;
}
QSplitter::handle:hover {
    background-color: #4fa3d1;
}

/* Table */
QTableWidget {
    background-color: #1e1e1e;
    gridline-color: #333;
    border: 1px solid #444;
    selection-background-color: #4fa3d1;
    selection-color: #1e1e1e;
}
QHeaderView::section {
    background-color: #3c3f41;
    color: #e0e0e0;
    padding: 4px;
    border: 1px solid #444;
}
"""
