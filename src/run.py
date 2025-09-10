#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Auto Gamma Analysis 애플리케이션 실행 파일
"""

import sys
from PyQt5.QtWidgets import QApplication
from main_app import GammaAnalysisApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GammaAnalysisApp()
    window.show()
    sys.exit(app.exec_())
