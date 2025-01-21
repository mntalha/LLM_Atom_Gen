import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict, Counter
from io import StringIO

# Jarvis imports
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.core.atoms import Atoms

# -------------------------------------------------------------------
# 1) Read CSV and parse CIFs
# -------------------------------------------------------------------
csv_file = "combined_structures_with_predictions_clean.csv"  # <-- change to your CSV file path
df_csv = pd.read_csv(csv_file)

# Prepare lists to store analysis results
elements_dict = defaultdict(int)
space_groups = []
prototypes = []
wyckoffs = []
wyckoffs_cell = []
crystal_systems = []
dimensionalities = []

# New lists for the additional properties you want to plot
alignn_mbj_bandgaps = []
alignn_formation_energies = []
alignn_ehulls = []

for idx, row in df_csv.iterrows():
    cif_str = row["gen_material_cif"]

    # Safely parse CIF from the string
    if pd.isna(cif_str) or not isinstance(cif_str, str) or len(cif_str.strip()) == 0:
        # Skip if empty or invalid
        continue

    fobj = StringIO(cif_str)
    try:
        # Load CIF into Jarvis Atoms (disable external 'cif2cell' by default)
        atoms = Atoms.from_cif(from_string=fobj.read(), use_cif2cell=False)
    except Exception as e:
        print(f"Could not parse CIF for row {idx}: {e}")
        continue

    # Try to get the space group
    try:
        spg = Spacegroup3D(atoms, symprec=1e-3)
        if spg._dataset is None:
            # spglib failed => force P1, triclinic
            print(f"Warning: spglib could not assign a space group for row {idx}. Using P1.")
            space_groups.append(1)  # P1
            crystal_systems.append("triclinic")
        else:
            # spglib succeeded
            space_groups.append(spg.space_group_number)
            crystal_systems.append(spg.crystal_system)
        
        # Wyckoff positions
        w = spg._dataset["wyckoffs"] if spg._dataset else []
        w2 = "".join(set(w))  # compressed label
        wyckoffs_cell.append(w2)
        for j in w:
            wyckoffs.append(j)

    except Exception as e:
        # If an unexpected error occurs in spglib
        print(f"Could not determine space group for row {idx}: {e}")
        # Force P1 or skip; here we force P1
        space_groups.append(1)
        crystal_systems.append("triclinic")
        wyckoffs_cell.append("")
        continue

    # Composition / prototype
    comp = atoms.composition
    prototypes.append(comp.prototype)

    # Collect atomic numbers
    for z in atoms.atomic_numbers:
        elements_dict[z] += 1

    # Dimensionality (if present)
    #dim = row.get("dimensionality", "na")
    #dimensionalities.append(dim)

    # Collect extra numeric properties
    alignn_mbj_bandgaps.append(row.get("alignn_mbj_bandgap", np.nan))
    alignn_formation_energies.append(row.get("alignn_formation_energy", np.nan))
    alignn_ehulls.append(row.get("alignn_ehull", np.nan))

# -------------------------------------------------------------------
# 2) Aggregate / sort data
# -------------------------------------------------------------------
max_items = 20000
sorted_proto = np.array(
    sorted(Counter(prototypes).items(), reverse=True, key=lambda x: x[1])[:max_items]
)
sorted_wyckoffs = np.array(
    sorted(Counter(wyckoffs).items(), reverse=True, key=lambda x: x[1])[:max_items]
)

dim_counts = Counter(dimensionalities)
dim_labels = list(dim_counts.keys())
dim_values = list(dim_counts.values())

# -------------------------------------------------------------------
# 3) Make plots
# -------------------------------------------------------------------
title_fontsize = 18
label_fontsize = 14
tick_fontsize = 12

with PdfPages("csv_cif_analysis.pdf") as pdf:
    fig, axs = plt.subplots(4, 3, figsize=(18, 20))
    # fig.suptitle("CSV+CIF Materials Analysis", fontsize=title_fontsize)

    # 0 -> (0,0),(0,1),(0,2)
    # 1 -> (1,0),(1,1),(1,2)
    # 2 -> (2,0),(2,1),(2,2)
    # 3 -> (3,0),(3,1),(3,2)

    # 1. Space group distribution (row=0, col=0)
    axs[0, 0].hist(space_groups, bins=30, alpha=0.7)
    axs[0, 0].set_title("Distribution of Space Group Numbers", fontsize=title_fontsize)
    axs[0, 0].set_xlabel("Space Group Number", fontsize=label_fontsize)
    axs[0, 0].set_ylabel("Frequency", fontsize=label_fontsize)
    axs[0, 0].tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 2. Atomic number distribution (row=0, col=1)
    axs[0, 1].bar(list(elements_dict.keys()), list(elements_dict.values()))
    axs[0, 1].set_title("Atomic Number Distribution", fontsize=title_fontsize)
    axs[0, 1].set_xlabel("Atomic Number", fontsize=label_fontsize)
    axs[0, 1].set_ylabel("Frequency", fontsize=label_fontsize)
    axs[0, 1].tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 3. Prototype distribution (row=0, col=2)
    if len(sorted_proto) > 0:
        axs[0, 2].bar(
            np.arange(len(sorted_proto)),
            sorted_proto[:, 1].astype(int),
            alpha=0.7
        )
        axs[0, 2].set_title("Prototype Distribution", fontsize=title_fontsize)
        axs[0, 2].set_xticks(np.arange(len(sorted_proto)))
        axs[0, 2].set_xticklabels(
            sorted_proto[:, 0],
            rotation=45,
            ha="right",
            fontsize=tick_fontsize
        )
        axs[0, 2].set_xlabel("Prototype", fontsize=label_fontsize)
        axs[0, 2].set_ylabel("Frequency", fontsize=label_fontsize)
    else:
        axs[0, 2].set_title("No Prototype Data", fontsize=title_fontsize)
    axs[0, 2].tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 4. Crystal system distribution (row=1, col=0)
    axs[1, 0].hist(crystal_systems, bins=np.arange(0, 5) - 0.5, alpha=0.7)
    axs[1, 0].set_title("Crystal System Distribution", fontsize=title_fontsize)
    axs[1, 0].set_xlabel("Crystal System (as integer label)", fontsize=label_fontsize)
    axs[1, 0].set_ylabel("Frequency", fontsize=label_fontsize)
    axs[1, 0].tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 5. Wyckoff site distribution (row=1, col=1)
    if len(sorted_wyckoffs) > 0:
        axs[1, 1].bar(
            np.arange(len(sorted_wyckoffs)),
            sorted_wyckoffs[:, 1].astype(int),
            alpha=0.7
        )
        axs[1, 1].set_title("Wyckoff Site Distribution", fontsize=title_fontsize)
        axs[1, 1].set_xticks(np.arange(len(sorted_wyckoffs)))
        axs[1, 1].set_xticklabels(
            sorted_wyckoffs[:, 0],
            fontsize=tick_fontsize
        )
        axs[1, 1].set_xlabel("Wyckoff Site", fontsize=label_fontsize)
        axs[1, 1].set_ylabel("Frequency", fontsize=label_fontsize)
    else:
        axs[1, 1].set_title("No Wyckoff Data", fontsize=title_fontsize)
    axs[1, 1].tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 6. ALIGNN MBJ Bandgap distribution (now at row=1, col=2)
    axs[1, 2].hist([val for val in alignn_mbj_bandgaps if not pd.isna(val)], bins=30, alpha=0.7)
    axs[1, 2].set_title("ALIGNN MBJ Bandgap Distribution", fontsize=title_fontsize)
    axs[1, 2].set_xlabel("Bandgap (eV)", fontsize=label_fontsize)
    axs[1, 2].set_ylabel("Frequency", fontsize=label_fontsize)
    axs[1, 2].tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 7. ALIGNN Formation Energy distribution (row=2, col=0)
    axs[2, 0].hist(
        [val for val in alignn_formation_energies if not pd.isna(val)],
        bins=30,
        alpha=0.7
    )
    axs[2, 0].set_title("ALIGNN Formation Energy Distribution", fontsize=title_fontsize)
    axs[2, 0].set_xlabel("Formation Energy (eV/atom)", fontsize=label_fontsize)
    axs[2, 0].set_ylabel("Frequency", fontsize=label_fontsize)
    axs[2, 0].tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 8. ALIGNN Energy Above Hull distribution (row=2, col=1)
    axs[2, 1].hist(
        [val for val in alignn_ehulls if not pd.isna(val)],
        bins=30,
        alpha=0.7
    )
    axs[2, 1].set_title("ALIGNN E_hull Distribution", fontsize=title_fontsize)
    axs[2, 1].set_xlabel("E_hull (eV/atom)", fontsize=label_fontsize)
    axs[2, 1].set_ylabel("Frequency", fontsize=label_fontsize)
    axs[2, 1].tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 9. Blank / extra (row=2, col=2)
    axs[2, 2].axis("off")

    # 10, 11, 12: last row all blank
    axs[3, 0].axis("off")
    axs[3, 1].axis("off")
    axs[3, 2].axis("off")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

