#!/usr/bin/env python3

import csv
import os
import re
import argparse
import subprocess

# ----------------------------------------------------------------------------
# User-adjustable settings
# ----------------------------------------------------------------------------
INPUT_CSV = "structures.csv"                     # Master input with all structures
PARTIAL_RESULTS_DIR = "partial_results"          # Where we'll store partial CSV outputs
OUTPUT_CIF_DIR = "cifs_local"                    # Temporary folder for CIF files
ALIGNN_PRETRAINED_PATH = "pretrained.py"  # Update to actual path
PROPERTIES = {
    "mbj_bandgap": "jv_mbj_bandgap_alignn",
    "formation_energy": "jv_formation_energy_peratom_alignn",
    "ehull": "jv_ehull_alignn"
}
# Regex for lines like:
#    Predicted value: jv_ehull_alignn test.cif [0.05731892213225365]
PATTERN = r"Predicted value:.*\[([-.\d]+)\]"



def run_alignn_on_cif(cif_path, model_name):
    """Call Alignn's pretrained model and parse the numeric output."""
    cmd = [
        ALIGNN_PRETRAINED_PATH,
        "--model_name", model_name,
        "--file_format", "cif",
        "--file_path", cif_path
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        match = re.search(PATTERN, cp.stdout)
        return float(match.group(1)) if match else None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Alignn failed on {cif_path} (model {model_name}): {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Alignn on a single CSV row.")
    parser.add_argument("--row_index", type=int, required=True,
                        help="1-based index of the row in structures.csv to process.")
    args = parser.parse_args()
    i = args.row_index  # 1-based

    # Read all rows from the master CSV
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Validate index
    if i < 1 or i > len(rows):
        raise ValueError(f"Row index {i} out of range (1..{len(rows)})")

    # Grab the row
    row = rows[i - 1]

    # Use a stable "entry_id" or fallback to "entry_{i}"
    # If the CSV has a unique field (like "material_id"), you can use that.
    entry_id = row.get("material_id", f"entry_{i}")

    # Ensure directories exist
    os.makedirs(OUTPUT_CIF_DIR, exist_ok=True)
    os.makedirs(PARTIAL_RESULTS_DIR, exist_ok=True)

    # Write the row's CIF
    cif_data = row.get("gen_material_cif", "")
    cif_path = os.path.join(OUTPUT_CIF_DIR, f"{entry_id}.cif")
    with open(cif_path, "w") as cif_file:
        cif_file.write(cif_data)

    # Run Alignn for each property
    mbj_val  = run_alignn_on_cif(cif_path, PROPERTIES["mbj_bandgap"])
    form_val = run_alignn_on_cif(cif_path, PROPERTIES["formation_energy"])
    ehull_val= run_alignn_on_cif(cif_path, PROPERTIES["ehull"])

    # Prepare partial CSV content
    # We'll store the row_index so we can merge later
    # plus optional "material_id" or any other ID, plus the new columns
    partial_header = [
        "row_index",
        "material_id",        # optional if your CSV uses another unique key
        "alignn_mbj_bandgap",
        "alignn_formation_energy",
        "alignn_ehull"
    ]
    partial_dict = {
        "row_index": str(i),
        "material_id": entry_id,
        "alignn_mbj_bandgap": mbj_val if mbj_val is not None else "",
        "alignn_formation_energy": form_val if form_val is not None else "",
        "alignn_ehull": ehull_val if ehull_val is not None else "",
    }

    # Write to partial_results/partial_i.csv
    out_csv_path = os.path.join(PARTIAL_RESULTS_DIR, f"partial_{i}.csv")
    with open(out_csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=partial_header)
        writer.writeheader()
        writer.writerow(partial_dict)

    print(f"[INFO] Finished row {i}; wrote {out_csv_path}")


if __name__ == "__main__":
    main()
