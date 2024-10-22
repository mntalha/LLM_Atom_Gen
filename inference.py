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
#os.environ['CUDA_VISIBLE_DEVICES']='0'
#torch.cuda.is_available = lambda : False
alpaca_prompt = """Below is a description of a superconductor material..

### Instruction:
{}

### Input:
{}

### Output:
{}"""

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  #
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/jipengsun/LLM_Atom_Gen/models/lora_model_m", # YOUR MODEL YOU USED FOR TRAINING models/llama-2-7b-bnb-4bit models/lora_model_m
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map="auto"
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
def text2atoms(response):
        tmp_atoms_array = response.split('\n')

        lat_lengths = np.array(tmp_atoms_array[1].split(),dtype='float')
        lat_angles = np.array(tmp_atoms_array[2].split(),dtype='float')

        lat = Lattice.from_parameters(lat_lengths[0], lat_lengths[1], lat_lengths[2], lat_angles[0], lat_angles[1], lat_angles[2])
        elements=[]
        coords=[]

        for ii,i in enumerate(tmp_atoms_array):
            if ii>2 and ii<len(tmp_atoms_array)-1 and ii % 2 == 1:  
                elements.append(i)
                tmp=(tmp_atoms_array[ii + 1].split())
                coords.append([float(tmp[0]),float(tmp[1]),float(tmp[2])])
        atoms = Atoms(coords=coords,elements=elements,lattice_mat=lat.lattice(),cartesian=False)
        return atoms

def gen_atoms(prompt="", max_new_tokens = 512):
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Below is a description of a superconductor material.", # instruction
                prompt, # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

        #for i in range(10):
        outputs = model.generate(**inputs, max_new_tokens = max_new_tokens, use_cache = True)
        response = tokenizer.batch_decode(outputs,
                                            skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=True)[0].split('# Output:')[1]
        print(tokenizer.batch_decode(outputs,
                                            skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=True)[0])
        atoms = text2atoms(response)

        return atoms

if __name__=="__main__":
 prompt_example = "The chemical formula is MgB2 The  Tc_supercon is 6.483. The spacegroup is 12. Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
#prompt_example = "The chemical formula is FeBN The  Tc_supercon is 36.483. Generate atomic structure description with lattice lengths, angles, coordinates and atom types."

 gen_mat = gen_atoms(prompt=prompt_example)
 print(gen_mat)
