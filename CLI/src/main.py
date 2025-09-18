"""
This module contains the main function for running the gamma analysis from the command line.
"""
import argparse
import yaml
import sys
import os

from file_handlers import DicomFileHandler, MCCFileHandler
from analysis import perform_gamma_analysis, extract_profile_data
from reporting import generate_report
from utils import logger, save_map_to_csv

def main():
    """Main function to run the gamma analysis from the command line."""
    parser = argparse.ArgumentParser(description="Run gamma analysis on DICOM RT and MCC files.")
    parser.add_argument("--rtplan", type=str, required=True, help="Path to the DICOM RT plan file.")
    parser.add_argument("--measure", type=str, required=True, help="Path to the measurement data file (MCC).")
    args = parser.parse_args()

    logger.info(f"Starting gamma analysis with the following arguments:")
    logger.info(f"  RT Plan: {args.rtplan}")
    logger.info(f"  Measurement: {args.measure}")

    # Load configuration
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        dta = config.get("dta", 3)
        dd = config.get("dd", 3)
        suppression_level = config.get("suppression_level", 10)
        save_csv = config.get("save_csv", False)
        csv_dir = config.get("csv_export_path", "csv_exports")
        logger.info(f"Loaded configuration: dta={dta}, dd={dd}, suppression_level={suppression_level}, save_csv={save_csv}, csv_export_path='{csv_dir}'")
    except FileNotFoundError:
        logger.error("Error: config.yaml not found. Please create it.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error: Could not parse config.yaml. Please check its format: {e}")
        sys.exit(1)

    # Initialize file handlers
    dicom_handler = DicomFileHandler()
    mcc_handler = MCCFileHandler()

    # Load files
    if not dicom_handler.open_file(args.rtplan):
        logger.error(f"Failed to load DICOM file: {args.rtplan}")
        sys.exit(1)

    if not mcc_handler.open_file(args.measure):
        logger.error(f"Failed to load MCC file: {args.measure}")
        sys.exit(1)

    # Crop MCC data to match the DICOM's calculated dose bounds
    if dicom_handler.dose_bounds:
        mcc_handler.crop_to_bounds(dicom_handler.dose_bounds)
        logger.info("Applied DICOM bounds to MCC data.")

    if save_csv:
        # Save ROI data to CSV in a dedicated directory
        os.makedirs(csv_dir, exist_ok=True)

        # Save DICOM ROI data
        dicom_basename = os.path.splitext(os.path.basename(args.rtplan))[0]
        dicom_csv_path = os.path.join(csv_dir, f"{dicom_basename}_dicom_roi.csv")
        save_map_to_csv(
            dicom_handler.get_pixel_data(),
            dicom_handler.phys_x_mesh,
            dicom_handler.phys_y_mesh,
            dicom_csv_path
        )

        # Save MCC ROI data
        mcc_basename = os.path.splitext(os.path.basename(args.measure))[0]
        mcc_csv_path = os.path.join(csv_dir, f"{mcc_basename}_mcc_roi.csv")
        save_map_to_csv(
            mcc_handler.get_pixel_data(),
            mcc_handler.phys_x_mesh,
            mcc_handler.phys_y_mesh,
            mcc_csv_path
        )

    # Perform gamma analysis
    try:
        gamma_map, gamma_stats, phys_extent, mcc_interp_data, dd_map, dta_map, dd_stats, dta_stats = perform_gamma_analysis(
            mcc_handler, dicom_handler,
            dd, dta,
            global_normalisation=True,
            threshold=suppression_level,
            max_gamma=None,
            save_csv=save_csv,
            csv_dir=csv_dir
        )

        if 'pass_rate' in gamma_stats:
            print("Gamma Analysis Results:")
            print(f"  Pass Rate: {gamma_stats['pass_rate']:.2f}%")
            print(f"  Mean Gamma: {gamma_stats['mean']:.3f}")
            print(f"  Max Gamma: {gamma_stats['max']:.3f}")
            print(f"  Min Gamma: {gamma_stats['min']:.3f}")
            print(f"  Total Points: {gamma_stats['total_points']}")

            # Generate profile data for the report (center and shifted profiles)
            ver_profile_data = extract_profile_data(
                direction="vertical",
                fixed_position=0,
                dicom_handler=dicom_handler,
                mcc_handler=mcc_handler
            )
            hor_profile_data = extract_profile_data(
                direction="horizontal",
                fixed_position=0,
                dicom_handler=dicom_handler,
                mcc_handler=mcc_handler
            )
            
            # Additional profiles shifted by ±1cm
            ver_profile_data_plus1 = extract_profile_data(
                direction="vertical",
                fixed_position=10,  # +1cm in mm
                dicom_handler=dicom_handler,
                mcc_handler=mcc_handler
            )
            ver_profile_data_minus1 = extract_profile_data(
                direction="vertical",
                fixed_position=-10,  # -1cm in mm
                dicom_handler=dicom_handler,
                mcc_handler=mcc_handler
            )
            hor_profile_data_plus1 = extract_profile_data(
                direction="horizontal",
                fixed_position=10,  # +1cm in mm
                dicom_handler=dicom_handler,
                mcc_handler=mcc_handler
            )
            hor_profile_data_minus1 = extract_profile_data(
                direction="horizontal",
                fixed_position=-10,  # -1cm in mm
                dicom_handler=dicom_handler,
                mcc_handler=mcc_handler
            )

            # Generate report
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            _, patient_id, _ = dicom_handler.get_patient_info()
            output_filename = os.path.join(reports_dir, f"report_{patient_id}_{dicom_handler.get_filename()}.jpg")
            generate_report(
                output_filename,
                dicom_handler,
                mcc_handler,
                gamma_map,
                gamma_stats,
                dta,
                dd,
                suppression_level,
                ver_profile_data,
                hor_profile_data,
                mcc_interp_data=mcc_interp_data,
                dd_map=dd_map,
                dta_map=dta_map,
                dd_stats=dd_stats,
                dta_stats=dta_stats,
                additional_profiles={
                    'ver_plus1': ver_profile_data_plus1,
                    'ver_minus1': ver_profile_data_minus1,
                    'hor_plus1': hor_profile_data_plus1,
                    'hor_minus1': hor_profile_data_minus1
                }
            )
            logger.info(f"Report saved to {output_filename}")

        else:
            print("Gamma analysis did not produce valid results.")

    except Exception as e:
        logger.error(f"An error occurred during gamma analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
