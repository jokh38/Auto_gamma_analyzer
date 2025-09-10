import numpy as np
from scipy.interpolate import interpn
from file_handlers import DicomFileHandler, MCCFileHandler

dicom_filepath = "1G240_2cm.dcm"
mcc_filepath = "1G240_2cm.mcc"

# DICOM 데이터 로드 및 (0,0)mm 지점의 선량 값 추출
dicom_handler = DicomFileHandler()
dicom_handler.open_file(dicom_filepath)
dicom_data = dicom_handler.get_pixel_data()
dicom_phys_y_coords = dicom_handler.phys_y_mesh[:, 0]
dicom_phys_x_coords = dicom_handler.phys_x_mesh[0, :]

# MCC 데이터 로드 및 (0,0)mm 지점의 선량 값 추출
mcc_handler = MCCFileHandler()
mcc_handler.open_file(mcc_filepath)
mcc_data = mcc_handler.get_matrix_data()
mcc_phys_y_coords = mcc_handler.phys_y_mesh[:, 0]
mcc_phys_x_coords = mcc_handler.phys_x_mesh[0, :]

# 물리적 좌표 (0,0)mm 지점
point_of_interest = np.array([[0.0, 0.0]]) # [[y, x]]

# DICOM 데이터에서 (0,0)mm 지점의 선량 값 보간
dicom_value_at_0_0_mm = interpn(
    (dicom_phys_y_coords, dicom_phys_x_coords),
    dicom_data,
    point_of_interest,
    method='linear',
    bounds_error=False,
    fill_value=np.nan
)[0]

# MCC 데이터에서 (0,0)mm 지점의 선량 값 보간
mcc_value_at_0_0_mm = interpn(
    (mcc_phys_y_coords, mcc_phys_x_coords),
    mcc_data,
    point_of_interest,
    method='linear',
    bounds_error=False,
    fill_value=np.nan
)[0]

print(f"DICOM 데이터 (0,0)mm 위치 선량 값: {dicom_value_at_0_0_mm}")
print(f"MCC 데이터 (0,0)mm 위치 선량 값: {mcc_value_at_0_0_mm}")
