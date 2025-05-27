# LLM_Atom_Gen

LLM_Atom_Gen is a framework designed to generate novel, non-existent atoms using Large Language Models (LLMs). This repository provides tools and scripts for setting up the environment, training, evaluation, and analysis for atom generation and materials discovery.

## Features

- **Environment Setup:** Reproducible Conda environment via `environment.yml`.
- **Training & Inference:** Python scripts and Jupyter notebooks for training, inference, and evaluation of LLMs for atom/material generation.
- **Data Analysis:** Notebooks for dataset exploration, baseline creation, and results comparison.
- **Research-Oriented:** Built for experimentation and extension in computational chemistry and AI-driven material discovery.

## Getting Started

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/mntalha/LLM_Atom_Gen.git
    cd LLM_Atom_Gen
    ```

2. **Create Conda Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate llm
    ```

3. **Run Experiments:**
    - Use the provided Python scripts and Jupyter notebooks for data preparation, model training, inference, and evaluation.

## Repository Structure

- `environment.yml` — Conda environment specification.
- `0_main.py`, `1_GEN_MbjBandgap.py`, `1_GEN_Tc_Supercon.py`, `1_sample_gen.py`, `2-sample_eval.py`, `inference.py`, `database_check.py`, `eval_functions.py`, `models.py`, etc. — Core Python scripts for training, generation, evaluation, and utilities.
- Jupyter Notebooks:
    - `(Plots1)ScatterPlots.ipynb`, `(Plots2)Database_Cmp.ipynb`, `(Plots3)Dataset_Exploaration.ipynb`, `(Plots4)figures.ipynb`
    - `ALIGNN_Extension.ipynb`, `BaselineCreation.ipynb`, `Database_Check.ipynb`, `CSV_File_Checker.ipynb`, `FinalComparison.ipynb`, `MaterialGen.ipynb`, `Mistral_Comp.ipynb`, `Reprocubility.ipynb`, `Relaxer_Result.ipynb`, `Tc_Supercon.ipynb`, etc.
- `data/` — (If present) Data files for training and evaluation.
- `0_models_tc_supercon/`, `1_models_optb88vdw_bandgap/`, `2_models_mbj_bandgap/` — Model directories (see their README files for download links).
- `.env.example` — Example environment variable file.
- `README.md` — Main documentation and usage instructions.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

Distributed under the MIT License. See `LICENSE` for more information.