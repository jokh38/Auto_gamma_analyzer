[33mcommit c377393275c11a53a540d572c5c6fcd36cb57b5a[m
Author: Kwanghyun Jo <jokh38@gmail.com>
Date:   Fri Sep 19 09:39:24 2025 +0900

    [BEHAVIORAL] Fix DICOM y-axis orientation and method name compatibility
    
    - Fix AttributeError: DicomFileHandler method name from create_physical_coordinates() to create_physical_coordinates_dcm()
    - Fix vertical profile y-coordinate sorting by moving sort_required outside mcc_handler conditional block
    - Ensures DICOM data y-axis is properly oriented for profile extraction and gamma analysis
    
    🤖 Generated with [Claude Code](https://claude.ai/code)
    
    Co-Authored-By: Claude <noreply@anthropic.com>

[1mdiff --git a/src/analysis.py b/src/analysis.py[m
[1mindex 2540233..54d1727 100644[m
[1m--- a/src/analysis.py[m
[1m+++ b/src/analysis.py[m
[36m@@ -35,21 +35,21 @@[m [mdef extract_profile_data(direction, fixed_position, dicom_handler, mcc_handler=N[m
             fixed_axis_coords_dicom = dicom_handler.phys_x_mesh[0, :][m
             profile_axis_mesh_dicom = dicom_handler.phys_y_mesh[m
             slicer_dicom = lambda idx: (slice(None), idx)[m
[32m+[m[32m            sort_required = True  # y-axis is inverted and needs sorting for interpolation[m
             if mcc_handler:[m
                 mcc_fixed_axis_coords = mcc_handler.phys_x_mesh[0, :][m
                 mcc_profile_axis_mesh = mcc_handler.phys_y_mesh[m
                 slicer_mcc = lambda idx: (slice(None), idx)[m
[31m-                sort_required = True  # MCC y-axis is inverted and needs sorting for interpolation[m
         else:  # "horizontal"[m
             # Horizontal profile: y is fixed, x is the profile axis[m
             fixed_axis_coords_dicom = dicom_handler.phys_y_mesh[:, 0][m
             profile_axis_mesh_dicom = dicom_handler.phys_x_mesh[m
             slicer_dicom = lambda idx: (idx, slice(None))[m
[32m+[m[32m            sort_required = False # x-axis is already sorted[m
             if mcc_handler:[m
                 mcc_fixed_axis_coords = mcc_handler.phys_y_mesh[:, 0][m
                 mcc_profile_axis_mesh = mcc_handler.phys_x_mesh[m
                 slicer_mcc = lambda idx: (idx, slice(None))[m
[31m-                sort_required = False # MCC x-axis is already sorted[m
 [m
         # --- Common Logic for both directions ---[m
 [m
[1mdiff --git a/src/main_app.py b/src/main_app.py[m
[1mindex 0b2815a..9eab786 100644[m
[1m--- a/src/main_app.py[m
[1m+++ b/src/main_app.py[m
[36m@@ -341,7 +341,7 @@[m [mclass GammaAnalysisApp(QMainWindow):[m
         """DICOM 원점 x좌표 업데이트"""[m
         if self.dicom_handler.get_pixel_data() is None: return[m
         self.dicom_handler.dicom_origin_x = self.dicom_x_spin.value()[m
[31m-        self.dicom_handler.create_physical_coordinates()[m
[32m+[m[32m        self.dicom_handler.create_physical_coordinates_dcm()[m
         self.dicom_origin_x, _ = self.dicom_handler.get_origin_coords()[m
         self.redraw_all_images()[m
     [m
[36m@@ -349,7 +349,7 @@[m [mclass GammaAnalysisApp(QMainWindow):[m
         """DICOM 원점 y좌표 업데이트"""[m
         if self.dicom_handler.get_pixel_data() is None: return[m
         self.dicom_handler.dicom_origin_y = self.dicom_y_spin.value()[m
[31m-        self.dicom_handler.create_physical_coordinates()[m
[32m+[m[32m        self.dicom_handler.create_physical_coordinates_dcm()[m
         _, self.dicom_origin_y = self.dicom_handler.get_origin_coords()[m
         self.redraw_all_images()[m
 [m
