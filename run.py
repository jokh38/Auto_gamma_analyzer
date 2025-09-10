#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Auto Gamma Analysis 애플리케이션 실행 파일
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

# src 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main_app import GammaAnalysisApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GammaAnalysisApp()
    window.show()
    sys.exit(app.exec_())
