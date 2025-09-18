"""
This module provides utility functions for the application, including logging and array manipulation.
"""
import os
import logging
import numpy as np
from datetime import datetime
import csv

# Check and create log directory
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Log file setup (daily log file)
log_file = os.path.join(log_dir, f'gamma_analysis_{datetime.now().strftime("%Y%m%d")}.log')

# Logger setup
def setup_logger(name):
    """Sets up a logger for the application.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # If handlers are already added, do not add them again
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(file_format)
        logger.addHandler(console_handler)
    
    return logger

# Create a default logger
logger = setup_logger('gamma_analysis')

def find_nearest_index(array, value):
    """Returns the index of the element in the array that is closest to the given value.

    Args:
        array (numpy.ndarray): The array to search.
        value (float): The value to find.

    Returns:
        int: The index of the nearest value.
    """
    return np.argmin(np.abs(array - value))

def save_map_to_csv(data_map, phys_x_mesh, phys_y_mesh, output_filename):
    """Saves a 2D data map with physical coordinates to a CSV file."""
    if data_map is None or phys_x_mesh is None or phys_y_mesh is None:
        logger.warning(f"Cannot save map to {output_filename}, data is not available.")
        return

    try:
        phys_x_coords = phys_x_mesh[0, :]
        phys_y_coords = phys_y_mesh[:, 0]
        height, _ = data_map.shape

        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header row (x-coordinates)
            header = ['y_mm \\ x_mm'] + [f"{x:.2f}" for x in phys_x_coords]
            writer.writerow(header)
            
            # Write data rows (y-coordinate + data)
            for i in range(height):
                row_data = [f"{val:.4f}" if not np.isnan(val) else "" for val in data_map[i, :]]
                row = [f"{phys_y_coords[i]:.2f}"] + row_data
                writer.writerow(row)

        logger.info(f"Map data saved to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save map to CSV {output_filename}: {e}", exc_info=True)
