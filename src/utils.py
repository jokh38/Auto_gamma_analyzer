"""
This module provides utility functions for the application, including logging and array manipulation.
"""
import os
import sys
import logging
import re
import numpy as np
from datetime import datetime
import csv
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_frozen_app():
    """Returns whether the app is running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def get_bundle_dir():
    """Returns the PyInstaller extraction dir or the repository root."""
    if is_frozen_app() and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return project_root


def get_user_data_dir():
    """Returns a writable per-user directory for logs and app output."""
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return os.path.join(local_appdata, "AutoGammaAnalyzer")
    return os.path.join(os.path.expanduser("~"), ".auto_gamma_analyzer")


def get_default_report_dir():
    """Returns the default directory for generated reports."""
    documents_dir = os.path.join(os.path.expanduser("~"), "Documents")
    if os.path.isdir(documents_dir):
        return os.path.join(documents_dir, "AutoGammaAnalyzer", "Report")
    return os.path.join(get_user_data_dir(), "Report")


log_dir = os.path.join(get_user_data_dir(), "logs")
os.makedirs(log_dir, exist_ok=True)

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
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        except Exception as exc:
            # Keep the application importable even if the log directory is not writable.
            logging.getLogger(__name__).warning(f"File logging disabled: {exc}")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(file_format)
        logger.addHandler(console_handler)
    
    return logger

# Create a default logger
logger = setup_logger('gamma_analysis')

DEFAULT_CONFIG = {
    "dta": 3,
    "dd": 3,
    "suppression_level": 10,
    "roi_margin": 2,
    "save_csv": False,
    "csv_export_path": "csv_exports",
    "interpolation_method": "cubic",
    "smoothing_factor": 1.0,
    "fill_value_type": "zero",
    "mcc_interpolation_method": "cubic",
    "mcc_fill_value_type": "zero",
    "roi_threshold_percent": 1,
}


def load_app_config():
    """Loads application config from the executable dir or bundled defaults."""
    config = DEFAULT_CONFIG.copy()
    config_paths = [
        os.path.join(os.path.dirname(sys.executable), "config.yaml")
        if is_frozen_app() else None,
        os.path.join(get_bundle_dir(), "config.yaml"),
        os.path.join(project_root, "config.yaml"),
    ]

    for config_path in [path for path in config_paths if path]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError("config.yaml must contain a top-level mapping.")
            config.update(loaded)
            logger.info(f"Loaded config from {config_path}")
            return config
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.error(f"Error loading config.yaml from {config_path}. Using default values: {e}")
            return config

    logger.warning("config.yaml not found. Using default values.")
    return config


def get_app_config_path():
    """Returns the most appropriate config.yaml path to read or update."""
    config_paths = [
        os.path.join(os.path.dirname(sys.executable), "config.yaml")
        if is_frozen_app() else None,
        os.path.join(get_bundle_dir(), "config.yaml"),
        os.path.join(project_root, "config.yaml"),
    ]

    for config_path in [path for path in config_paths if path]:
        if os.path.isfile(config_path):
            if os.access(os.path.dirname(config_path) or ".", os.W_OK):
                return config_path

    for config_path in [path for path in config_paths if path]:
        if os.access(os.path.dirname(config_path) or ".", os.W_OK):
            return config_path

    return os.path.join(project_root, "config.yaml")


def update_app_config_value(key, value):
    """Updates a single top-level config entry while preserving file formatting."""
    config_path = get_app_config_path()
    value_text = str(value)

    try:
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                contents = f.read()

            pattern = rf"(?m)^({re.escape(key)}:\s*)(.*)$"
            replacement = rf"\g<1>{value_text}"
            updated_contents, count = re.subn(pattern, replacement, contents, count=1)

            if count == 0:
                updated_contents = contents.rstrip() + f"\n{key}: {value_text}\n"

            with open(config_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(updated_contents)
        else:
            config = load_app_config()
            config[key] = value
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

        logger.info(f"Updated {key} in {config_path} to {value_text}")
        return config_path
    except Exception as exc:
        logger.error(f"Failed to update {key} in config.yaml: {exc}", exc_info=True)
        raise


def get_config_fill_value(fill_value_type):
    """Converts fill value config to the numeric value used by interpolation."""
    return 0.0 if fill_value_type == "zero" else np.nan

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
