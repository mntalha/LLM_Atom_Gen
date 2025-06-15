## SemiconLLM: Evaluating Large Language Models for Inverse Semiconductor Design

A project for generating atomic structures using Large Language Models (LLMs).

In this study, we fine-tune multiple LLMs on various density functional theory (DFT) datasets (including superconducting and semiconducting materials at different levels of DFT theory) and apply quantitative metrics to benchmark their effectiveness.

![Example Atomic Structure](figures/FullPipeline.png)

## Folder Structure

```
LLM_Atom_Gen/
├── data/
├── figures/
├── analysis/
├── notebooks/
├── outputs/
├── colabnotebooks/
├── *.py
├── README.md
├── requirements.txt
├── run_model.sh
```

## Colab Notebook Example

You can find an example Colab notebook demonstrating the Mistral model generating an atomic structure in [`colabnotebooks/MistralExampleGeneration.ipynb`](colabnotebooks/MistralExampleGeneration.ipynb).


## Setting Up the Environment

You can set up the project dependencies using either `requirements.txt` or `environment.yaml`:

### Using `requirements.txt` (with pip)

```bash
pip install -r requirements.txt
```

### Colab Notebooks

The `colabnotebooks/` folder contains example notebooks that can be easily run on Google Colab. These notebooks provide step-by-step demonstrations for data processing, model training, and inference, making it convenient to experiment with the project in a cloud environment without local setup.
Currently available notebook(s):

- `MistralExampleGeneration.ipynb`: Demonstrates example generation using the Mistral model on Colab, including library installation and model inference.

### Using `environment.yaml` (with conda)

```bash
conda env create -f environment.yaml
conda activate <your_env_name>
```

Replace `<your_env_name>` with the name specified in the `environment.yaml` file.

## Running Model Training

Before running the training script, ensure it is executable. You can set the execute permission with:

```bash
chmod +x run_model.sh
```

To train the model using the provided shell script, run:

```bash
./run_model.sh
```

This script will execute the necessary commands to start the training process. Make sure you have configured any required arguments or environment variables inside `run_model.sh` before running it.

## Model Selection

Model training is managed by `main.py`, which takes an argument specifying the model to use. The available models are indexed as follows:

| Index | Model Name                                      |
|-------|-------------------------------------------------|
| 0     | unsloth/tinyllama-chat                          |
| 1     | unsloth/mistral-7b-bnb-4bit                     |
| 2     | unsloth/mistral-7b-instruct-v0.2-bnb-4bit       |
| 3     | unsloth/llama-2-7b-bnb-4bit                     |
| 4     | unsloth/gemma-7b-bnb-4bit                       |
| 5     | unsloth/llama-3-8b-bnb-4bit                     |
| 6     | unsloth/llama-2-13b-bnb-4bit                    |
| 7     | unsloth/codellama-34b-bnb-4bit                  |
| 8     | unsloth/llama-3-70b-bnb-4bit                    |
| 9     | knc6/atomgpt_mistral_tc_supercon                |

To train with a specific model, run:

```bash
python main.py --model <index>
```

Replace `<index>` with the desired model's index from the table above.

## Inference

To perform inference using a trained model, refer to `generation.py`. The script demonstrates inference for the `tc_supercon` dataset, but is structured to be generalizable for other datasets as well.

Note: Input and output files are not specified via command-line arguments; instead, data paths and output locations are defined within the code itself. To adapt `generation.py` for different datasets, modify the relevant data loading and preprocessing sections directly in the script.

## Mistral Model Saved Locations

You can access the saved Mistral model from the following sources:

| Platform        | Link                                                                                  |
|-----------------|---------------------------------------------------------------------------------------|
| **Hugging Face**| [Mistral-Model on Hugging Face](https://huggingface.co/mntalha/Mistral-Model)         |
| **Google Drive**| [Mistral-Model on Google Drive](https://drive.google.com/drive/folders/1_gjPU-N7rOz09fS7PVR4hAWyzu-HUp8D?usp=share_link) |



## Model Time Comparison

To compare the inference or training times of different models, use the `time_cmp.py` script. This script benchmarks the performance of each model listed above and outputs timing statistics.

### Running Time Comparison

```bash
python time_cmp.py
```

## Publication

Muhammed Nur Talha Kilic, Daniel Wines, Kamal Choudhary,Vishu Gupta, Youjia Li, Sayak Chakrabarty, Wei-Keng Liao, Alok Choudhary, Ankit Agrawal", Manuscript in preparation.


## Contact
Ankit Agrawal <ankit-agrawal@northwestern.edu> and Talha Kilic <talha.kilic@u.northwestern.edu>

Copyright (C) 2025, Northwestern University.

See COPYRIGHT notice in top-level directory.


## Funding Support
This research was supported by the National Science Foundation (NSF) CMMI Division Grant CMMI-2053840/2053929. Partial support from NIST award 70NANB19H005 and Northwestern Center for Nanocombinatorics is also acknowledged.

<!-- 
    Documentation generated by GitHub Copilot and other LLM models.
    This documentation comment provides an overview and usage details for the selected code.
    For more information, refer to the project README file.
-->
