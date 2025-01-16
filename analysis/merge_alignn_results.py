#!/usr/bin/env python3

import csv
import os
import glob

# ------------------------------------------------------------------------
# User-adjustable
# ------------------------------------------------------------------------
MASTER_CSV = "structures.csv"
PARTIAL_RESULTS_DIR = "partial_results"
OUTPUT_CSV = "structures_with_predictions.csv"


def main():
    # 1) Read the master CSV
    with open(MASTER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Prepare a dictionary keyed by row_index => entire row
    # We'll store them as 1-based indexing, matching Slurm tasks
    row_dict = {}
    for i, row in enumerate(rows, start=1):
        row_dict[i] = row

    # 2) Read partial CSV files
    partial_paths = sorted(glob.glob(os.path.join(PARTIAL_RESULTS_DIR, "partial_*.csv")))
    print(f"[INFO] Found {len(partial_paths)} partial CSV files.")
    
    # For each partial file, we expect exactly one row with columns like:
    # row_index, material_id, alignn_mbj_bandgap, ...
    for ppath in partial_paths:
        with open(ppath, "r", encoding="utf-8") as fin:
            preader = csv.DictReader(fin)
            partial_rows = list(preader)
            # Usually partial_rows has exactly 1 row, if everything is correct
            if len(partial_rows) != 1:
                print(f"[WARNING] {ppath} has {len(partial_rows)} rows, expected 1.")
                continue
            prow = partial_rows[0]
        
        # Extract the row_index from prow
        try:
            idx = int(prow["row_index"])
        except Exception as e:
            print(f"[ERROR] Could not parse row_index in {ppath}: {e}")
            continue

        # Now update the main row in row_dict
        if idx in row_dict:
            row_dict[idx]["alignn_mbj_bandgap"] = prow.get("alignn_mbj_bandgap", "")
            row_dict[idx]["alignn_formation_energy"] = prow.get("alignn_formation_energy", "")
            row_dict[idx]["alignn_ehull"] = prow.get("alignn_ehull", "")

    # 3) Determine final fieldnames
    # Take the original columns from the master CSV and add the new columns
    orig_fieldnames = list(rows[0].keys())
    extra_cols = ["alignn_mbj_bandgap", "alignn_formation_energy", "alignn_ehull"]
    # If the original CSV already has these columns, no duplication
    final_fieldnames = orig_fieldnames[:]
    for col in extra_cols:
        if col not in final_fieldnames:
            final_fieldnames.append(col)

    # 4) Write out the final CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=final_fieldnames)
        writer.writeheader()
        # row_dict keys are 1..N, so we'll iterate in sorted order
        for i in range(1, len(rows)+1):
            writer.writerow(row_dict[i])

    print(f"[INFO] Wrote final merged predictions => {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
