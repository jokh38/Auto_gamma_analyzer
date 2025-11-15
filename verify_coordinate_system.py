#!/usr/bin/env python3
"""
Verification script for coordinate system changes.
Checks that all imshow calls use origin='upper' and MCC Y-axis transformation is consistent.
"""

import re
import sys

def check_file(filepath, pattern, expected, description):
    """Check if a file contains expected pattern."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            matches = re.findall(pattern, content)

            if not matches:
                print(f"✗ {description}: No matches found in {filepath}")
                return False

            all_correct = all(expected in match for match in matches)

            if all_correct:
                print(f"✓ {description}: All {len(matches)} instances correct in {filepath}")
                return True
            else:
                print(f"✗ {description}: Some instances incorrect in {filepath}")
                for match in matches:
                    if expected not in match:
                        print(f"  Found: {match}")
                return False
    except Exception as e:
        print(f"✗ Error checking {filepath}: {e}")
        return False

def main():
    print("Verifying coordinate system changes...\n")

    all_passed = True

    # Check 1: All imshow calls should have origin='upper'
    files_to_check = [
        'src/ui_components.py',
        'src/reporting.py'
    ]

    for filepath in files_to_check:
        result = check_file(
            filepath,
            r"\.imshow\([^)]*origin=['\"](\w+)['\"]",
            'upper',
            f"imshow origin parameter"
        )
        all_passed = all_passed and result

    print()

    # Check 2: MCC Y-axis transformation should NOT have negative sign
    result = check_file(
        'src/file_handlers.py',
        r"phys_y = ([^=]*full_grid_py[^=]*mcc_origin_y[^=]*mcc_spacing_y)",
        '(full_grid_py - self.mcc_origin_y) * self.mcc_spacing_y',
        "MCC Y-axis transformation (file_handlers.py)"
    )
    all_passed = all_passed and result

    result = check_file(
        'src/analysis.py',
        r"phys_y_all = ([^=]*full_grid_py[^=]*mcc_origin_y[^=]*mcc_spacing_y)",
        '(full_grid_py - handler.mcc_origin_y) * handler.mcc_spacing_y',
        "MCC Y-axis transformation (analysis.py)"
    )
    all_passed = all_passed and result

    print("\n" + "="*60)
    if all_passed:
        print("✓ All coordinate system changes verified successfully!")
        print("="*60)
        return 0
    else:
        print("✗ Some verification checks failed!")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
