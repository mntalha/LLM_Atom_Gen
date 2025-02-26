#!/usr/bin/env python3
"""
This script reads a CSV file with columns:
    - formula
    - gen_material_cif
It queries several OPTIMADE databases for matching structures, converts 
the returned OPTIMADE structure resources to pymatgen Structures, compares 
them to the generated structure (from the CIF), and then saves organized 
results into a CSV with columns for each database.
"""

import re
import pandas as pd

# OPTIMADE Client and converters
from optimade.client import OptimadeClient
from optimade.adapters.structures.pymatgen import get_pymatgen
from optimade.adapters import Structure as OptimadeStructure

# Pymatgen libraries
from pymatgen.core import Structure as PMGStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher


def swap_formula(formula: str) -> str:
    """Return the formula with the element order reversed."""
    elements = re.findall(r"[A-Z][a-z]?\d*", formula)
    return "".join(reversed(elements))


def patch_structure_dict(structure_dict: dict) -> dict:
    """Patch the raw OPTIMADE structure dictionary to fill in missing fields,
    especially for OQMD.
    """
    # Ensure "id" is a string.
    if not isinstance(structure_dict.get("id"), str):
        structure_dict["id"] = str(structure_dict.get("id", ""))
    # Get the attributes dict (or empty dict if missing).
    attributes = structure_dict.get("attributes", {})
    # If "dimension_types" is missing but "nperiodic_dimensions" is present, supply a default.
    if "nperiodic_dimensions" in attributes and "dimension_types" not in attributes:
        attributes["dimension_types"] = [1] * attributes["nperiodic_dimensions"]
    # If "species" is missing but "species_at_sites" is present, construct a minimal species list.
    if "species_at_sites" in attributes and "species" not in attributes:
        unique_species = list(set(attributes["species_at_sites"]))
        species_list = []
        for el in unique_species:
            species_list.append({
                "name": el,
                "chemical_symbols": [el],
                "concentration": [1.0]
            })
        attributes["species"] = species_list
    structure_dict["attributes"] = attributes
    return structure_dict


def process_database(client: OptimadeClient, db_name: str, base_url: str, formula: str, gen_structure, matcher: StructureMatcher):
    """Query a database for a formula and process the results.
    Returns a dictionary with keys for this database.
    """
    results = {
        f"{db_name}_nentries": 0,
        f"{db_name}_matching_ids": "",
        f"{db_name}_first_id": "",
        f"{db_name}_first_sg_symbol": "",
        f"{db_name}_first_sg_number": "",
        f"{db_name}_lattice_a": "",
        f"{db_name}_lattice_b": "",
        f"{db_name}_lattice_c": "",
        f"{db_name}_lattice_alpha": "",
        f"{db_name}_lattice_beta": "",
        f"{db_name}_lattice_gamma": "",
    }
    # Build the query for the given formula
    query = f'chemical_formula_reduced="{formula}"'
    response = client.get(filter=query)
    structures = response["structures"][query][base_url]["data"]

    # If no results, try the swapped formula
    if not structures:
        swapped = swap_formula(formula)
        query = f'chemical_formula_reduced="{swapped}"'
        response = client.get(filter=query)
        structures = response["structures"][query][base_url]["data"]

    results[f"{db_name}_nentries"] = len(structures)

    if len(structures) == 0:
        return results

    # For each returned structure, patch and convert it, then test for a match.
    matching_ids = []
    first_match_data = None
    for struct in structures:
        patched = patch_structure_dict(struct)
        try:
            # Wrap in an OptimadeStructure to ensure proper attributes
            opt_struct = OptimadeStructure(patched)
            db_structure = get_pymatgen(opt_struct)
        except Exception as e:
            print(f"    -> ERROR converting structure {patched.get('id','N/A')}: {e}")
            continue

        # Check if this candidate matches the generated structure
        try:
            if matcher.fit(gen_structure, db_structure):
                matching_ids.append(patched.get("id", ""))
                # Save first match's info if not already saved
                if first_match_data is None:
                    sga_db = SpacegroupAnalyzer(db_structure, symprec=1e-3)
                    first_match_data = {
                        "id": patched.get("id", ""),
                        "sg_symbol": sga_db.get_space_group_symbol(),
                        "sg_number": sga_db.get_space_group_number(),
                        "lattice": db_structure.lattice
                    }
        except Exception as e:
            print(f"    -> ERROR matching structure {patched.get('id','N/A')}: {e}")

    results[f"{db_name}_matching_ids"] = ",".join(matching_ids)
    if first_match_data is not None:
        results[f"{db_name}_first_id"] = first_match_data["id"]
        results[f"{db_name}_first_sg_symbol"] = first_match_data["sg_symbol"]
        results[f"{db_name}_first_sg_number"] = first_match_data["sg_number"]
        lat = first_match_data["lattice"]
        results[f"{db_name}_lattice_a"] = lat.lengths[0]
        results[f"{db_name}_lattice_b"] = lat.lengths[1]
        results[f"{db_name}_lattice_c"] = lat.lengths[2]
        results[f"{db_name}_lattice_alpha"] = lat.angles[0]
        results[f"{db_name}_lattice_beta"] = lat.angles[1]
        results[f"{db_name}_lattice_gamma"] = lat.angles[2]
    return results


def main():
    # Read input CSV
    input_csv = "filtered.csv"
    df_input = pd.read_csv(input_csv)

    # List of databases to query
    databases = {
        "Alexandria": "https://alexandria.icams.rub.de/pbesol",
        "MC3D": "https://aiida.materialscloud.org/mc3d/optimade",
        "OQMD": "https://oqmd.org/optimade/",
        "JARVIS": "https://jarvis.nist.gov/optimade/jarvisdft",
        "MaterialsProject": "https://optimade.materialsproject.org/",
    }

    # Prepare the structure matcher.
    matcher = StructureMatcher(
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
        stol=0.5,
        angle_tol=5.0
    )

    # We'll build a list of result dictionaries.
    results_list = []
    for idx, row in df_input.iterrows():
        row_dict = {}
        formula = str(row["formula"])
        row_dict["formula"] = formula
        row_dict["gen_material_cif"] = row["gen_material_cif"]

        # 1) Convert the generated CIF to a pymatgen Structure.
        try:
            gen_structure = PMGStructure.from_str(row["gen_material_cif"], fmt="cif")
            sga_gen = SpacegroupAnalyzer(gen_structure, symprec=1e-3)
            row_dict["generated_sg_symbol"] = sga_gen.get_space_group_symbol()
            row_dict["generated_sg_number"] = sga_gen.get_space_group_number()
        except Exception as e:
            row_dict["generated_sg_symbol"] = ""
            row_dict["generated_sg_number"] = ""
            print(f" -> ERROR parsing generated CIF at row {idx}: {e}")
            results_list.append(row_dict)
            continue

        # For each database, create an OptimadeClient and process query.
        for db_name, base_url in databases.items():
            try:
                client = OptimadeClient(base_urls=[base_url])
                db_results = process_database(client, db_name, base_url, formula, gen_structure, matcher)
                row_dict.update(db_results)
            except Exception as e:
                print(f" -> ERROR processing {db_name} for row {idx} ({formula}): {e}")
                # Fill in empty values for this database if error
                row_dict[f"{db_name}_nentries"] = 0
                row_dict[f"{db_name}_matching_ids"] = ""
                row_dict[f"{db_name}_first_id"] = ""
                row_dict[f"{db_name}_first_sg_symbol"] = ""
                row_dict[f"{db_name}_first_sg_number"] = ""
                row_dict[f"{db_name}_lattice_a"] = ""
                row_dict[f"{db_name}_lattice_b"] = ""
                row_dict[f"{db_name}_lattice_c"] = ""
                row_dict[f"{db_name}_lattice_alpha"] = ""
                row_dict[f"{db_name}_lattice_beta"] = ""
                row_dict[f"{db_name}_lattice_gamma"] = ""
        results_list.append(row_dict)
        print(f" -> Finished processing row {idx} ({formula}).")

    # Create a DataFrame from the results and save to CSV.
    df_results = pd.DataFrame(results_list)
    output_csv = "filtered_add.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nDone! Final organized CSV saved to {output_csv}")


if __name__ == "__main__":
    main()
