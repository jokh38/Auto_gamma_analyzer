import numpy as np
from file_handlers import DicomFileHandler, MCCFileHandler

dicom_filepath = "1G240_2cm.dcm"
mcc_filepath = "1G240_2cm.mcc"

dicom_handler = DicomFileHandler()
dicom_handler.open_file(dicom_filepath)
dicom_data = dicom_handler.get_pixel_data()

mcc_handler = MCCFileHandler()
mcc_handler.open_file(mcc_filepath)
mcc_data = mcc_handler.get_matrix_data()

# (0,0) 위치의 값 확인
# numpy 배열은 [row, column] 또는 [y, x] 순서입니다.
# DICOM 데이터는 일반적으로 (y, x) 순서로 저장됩니다.
# MCC 데이터도 일반적으로 (y, x) 순서로 저장됩니다.

dicom_value_at_0_0 = dicom_data[0, 0]
mcc_value_at_0_0 = mcc_data[0, 0]

print(f"DICOM 데이터 (0,0) 위치 값: {dicom_value_at_0_0}")
print(f"MCC 데이터 (0,0) 위치 값: {mcc_value_at_0_0}")