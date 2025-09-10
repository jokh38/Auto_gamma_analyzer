import pydicom
ds = pydicom.dcmread('1G240_2cm.dcm')
print(f'ImagePositionPatient (0020,0032): {ds.ImagePositionPatient}')
