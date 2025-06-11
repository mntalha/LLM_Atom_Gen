from jarvis.db.jsonutils import loadjson
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson, dumpjson
import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice
from tqdm import tqdm
from jarvis.io.vasp.inputs import Poscar

import os
# Uncomment below to set a specific GPU device
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# Uncomment below to force CPU usage (for debugging)
# torch.cuda.is_available = lambda : False

# Template for the prompt given to the language model
alpaca_prompt = """Below is a description of a superconductor material..

### Instruction:
{}

### Input:
{}

### Output:
{}"""

# Model configuration
max_seq_length = 2048  # Maximum sequence length for the model
dtype = None           # Data type (None lets the model decide)
load_in_4bit = True    # Use 4-bit quantization for efficiency

# Load the fine-tuned language model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/models/lora_model_m", # Path to your trained model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map="auto"  # Automatically select device (GPU/CPU)
)
# Enable faster inference mode
FastLanguageModel.for_inference(model)

def text2atoms(response):
    """
    Converts the model's text output into a JARVIS Atoms object.
    Expects the response to have:
    - Lattice lengths on line 2
    - Lattice angles on line 3
    - Atom types and coordinates on subsequent lines
    """
    tmp_atoms_array = response.split('\n')

    # Parse lattice parameters
    lat_lengths = np.array(tmp_atoms_array[1].split(), dtype='float')
    lat_angles = np.array(tmp_atoms_array[2].split(), dtype='float')

    # Create lattice object from parameters
    lat = Lattice.from_parameters(
        lat_lengths[0], lat_lengths[1], lat_lengths[2],
        lat_angles[0], lat_angles[1], lat_angles[2]
    )
    elements = []
    coords = []

    # Parse atom types and coordinates
    for ii, i in enumerate(tmp_atoms_array):
        # Atom type lines are odd-indexed after the first two lines
        if ii > 2 and ii < len(tmp_atoms_array) - 1 and ii % 2 == 1:
            elements.append(i)
            tmp = tmp_atoms_array[ii + 1].split()
            coords.append([float(tmp[0]), float(tmp[1]), float(tmp[2])])
    # Create Atoms object
    atoms = Atoms(coords=coords, elements=elements, lattice_mat=lat.lattice(), cartesian=False)
    return atoms

def gen_atoms(prompt="", max_new_tokens=512):
    """
    Generates atomic structure from a natural language prompt using the language model.
    """
    # Format the prompt for the model
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Below is a description of a superconductor material.", # instruction
                prompt, # input
                "",     # output - left blank for generation
            )
        ], return_tensors="pt"
    ).to("cuda")  # Move tensors to GPU

    # Generate output from the model
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True
    )
    # Decode the output and extract the relevant part
    response = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].split('# Output:')[1]
    # Print the full decoded output for inspection
    print(tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0])
    # Convert text output to Atoms object
    atoms = text2atoms(response)

    return atoms

if __name__ == "__main__":
    # Example prompt describing a superconductor material
    prompt_example = (
        "The chemical formula is MgB2 The  Tc_supercon is 6.483. "
        "The spacegroup is 12. Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
    )
    # Alternative prompt example (commented out)
    # prompt_example = "The chemical formula is FeBN The  Tc_supercon is 36.483. Generate atomic structure description with lattice lengths, angles, coordinates and atom types."

    # Generate atomic structure from the prompt
    gen_mat = gen_atoms(prompt=prompt_example)
    print(gen_mat)
