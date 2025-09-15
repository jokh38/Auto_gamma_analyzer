import unittest
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from analysis import perform_gamma_analysis
from file_handlers import DicomFileHandler, MCCFileHandler

class TestGammaAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up the test case by loading the data files."""
        self.dicom_path = 'data/1G240_2cm.dcm'
        self.mcc_path = 'data/1G240_2cm.mcc'

        self.dicom_handler = DicomFileHandler()
        self.mcc_handler = MCCFileHandler()

        self.assertTrue(self.dicom_handler.open_file(self.dicom_path))
        self.assertTrue(self.mcc_handler.open_file(self.mcc_path))

    def test_perform_gamma_analysis(self):
        """
        Test the perform_gamma_analysis function with example data.
        """
        gamma_map, gamma_stats, phys_extent = perform_gamma_analysis(
            reference_handler=self.mcc_handler,
            evaluation_handler=self.dicom_handler,
            dose_percent_threshold=3,
            distance_mm_threshold=3
        )

        # Check that the gamma analysis produced results
        self.assertIsNotNone(gamma_map)
        self.assertIsNotNone(gamma_stats)
        self.assertIsNotNone(phys_extent)

        # Check the statistics
        self.assertIn('pass_rate', gamma_stats)
        self.assertIn('mean', gamma_stats)
        self.assertIn('max', gamma_stats)
        self.assertIn('min', gamma_stats)
        self.assertIn('total_points', gamma_stats)

        # Check that the pass rate is a float between 0 and 100
        self.assertIsInstance(gamma_stats['pass_rate'], float)
        self.assertGreaterEqual(gamma_stats['pass_rate'], 0)
        self.assertLessEqual(gamma_stats['pass_rate'], 100)

        # Check that the number of points is greater than 0
        self.assertGreater(gamma_stats['total_points'], 0)

if __name__ == '__main__':
    unittest.main()
