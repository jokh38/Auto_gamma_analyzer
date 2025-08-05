import argparse
import json
import sys

from file_handlers import DicomFileHandler, MCCFileHandler
from analysis import perform_gamma_analysis, extract_profile_data
from reporting import generate_report
from utils import logger

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
        with open("config.json", "r") as f:
            config = json.load(f)
        dta = config.get("dta", 3)
        dd = config.get("dd", 3)
        suppression_level = config.get("suppression_level", 10)
        logger.info(f"Loaded configuration: dta={dta}, dd={dd}, suppression_level={suppression_level}")
    except FileNotFoundError:
        logger.error("Error: config.json not found. Please create it.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Error: Could not decode config.json. Please check its format.")
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

    # Perform gamma analysis
    try:
        gamma_map, gamma_stats, phys_extent = perform_gamma_analysis(
            mcc_handler, dicom_handler,
            dd, dta,
            global_normalisation=True  # Assuming global gamma, can be made configurable
        )

        if 'pass_rate' in gamma_stats:
            print("Gamma Analysis Results:")
            print(f"  Pass Rate: {gamma_stats['pass_rate']:.2f}%")
            print(f"  Mean Gamma: {gamma_stats['mean']:.3f}")
            print(f"  Max Gamma: {gamma_stats['max']:.3f}")
            print(f"  Min Gamma: {gamma_stats['min']:.3f}")
            print(f"  Total Points: {gamma_stats['total_points']}")

            # Generate profile data for the report
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

            # Generate report
            _, patient_id, _ = dicom_handler.get_patient_info()
            output_filename = f"report_{patient_id}.jpg"
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
                hor_profile_data
            )
            logger.info(f"Report saved to {output_filename}")

        else:
            print("Gamma analysis did not produce valid results.")

    except Exception as e:
        logger.error(f"An error occurred during gamma analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
