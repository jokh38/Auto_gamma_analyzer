import os
import logging
import numpy as np
from datetime import datetime

# 로그 디렉토리 확인 및 생성
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 로그 파일 설정 (일별 로그 파일)
log_file = os.path.join(log_dir, f'gamma_analysis_{datetime.now().strftime("%Y%m%d")}.log')

# 로거 설정
def setup_logger(name):
    """애플리케이션용 로거 설정

    Args:
        name (str): 로거 이름

    Returns:
        logging.Logger: 설정된 로거 객체
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 이미 핸들러가 추가되어 있다면 추가하지 않음
    if not logger.handlers:
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(file_format)
        logger.addHandler(console_handler)
    
    return logger

# 기본 로거 생성
logger = setup_logger('gamma_analysis')

def find_nearest_index(array, value):
    """배열에서 주어진 값과 가장 가까운 요소의 인덱스 반환

    Args:
        array (numpy.ndarray): 탐색할 배열
        value (float): 찾을 값

    Returns:
        int: 가장 가까운 값의 인덱스
    """
    return np.argmin(np.abs(array - value))
