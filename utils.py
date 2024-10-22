#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:21:49 2024

@author: talha
"""

from jarvis.core.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice


def get_crystal_string_t(atoms):
    lengths = atoms.lattice.abc  # structure.lattice.parameters[:3]
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords

    # crystal_str = (
    #     " ".join(["{0:.2f}".format(x) for x in lengths])
    #     + "\n"
    #     + " ".join([str(int(x)) for x in angles])
    #     + "\n"
    #     + "\n".join(
    #         [
    #             str(t) + " " + " ".join(["{0:.3f}".format(x) for x in c])
    #             for t, c in zip(atom_ids, frac_coords)
    #         ]
    #     )
    # )
    crystal_str = \
    " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
    " ".join([str(int(x)) for x in angles]) + "\n" + \
    "\n".join([
        str(t) + "\n" + " ".join([
            "{0:.2f}".format(x) for x in c
        ]) for t,c in zip(atom_ids, frac_coords)
    ])

    # crystal_str = atoms_describer(atoms) + "\n*\n" + crystal_str
    return crystal_str


def make_alpaca_json(dataset=[], prop="Tc_supercon"):
    mem = []
    for i in dataset:
        if i[prop] != "na":  #i[prop] != "na"  i[prop] > 0.0
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            info["instruction"] = (
                "Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
            )
            info["input"] = (
                "The chemical formula is "
                + atoms.composition.reduced_formula
                + ". The  "
                + prop
                + " value is "
                + str(round(i[prop], 3))
                # + ". The spacegroup is "
                # + i["spg_number"]
                + "."
            )
            info["response"] = get_crystal_string_t(atoms)
            mem.append(info)
    return mem

def parse_fn(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]
    
    structure = Structure(
        lattice=Lattice.from_parameters(
            *(lengths + angles)),
        species=species,
        coords=coords, 
        coords_are_cartesian=False,
    )
    
    return structure.to(fmt="cif")


alpaca_prompt = """Below is a description of a superconductor material. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
        
#import re   
#output = re.sub(r'\S+', 'the', output)
    
def formatting_prompts_func(examples, tokenizer):

    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["response"]
    texts = []
    for instruction, input, output in zip(instructions,inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)

    return {
        "text": texts,
    }

def eval_prompts(examples):
    
    instructions = examples["instruction"]
    inputs = examples["input"]
    output = ""
    texts = []
    for instruction, input in zip(instructions, inputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) #+'</s>' #tokenizer.eos_token #'</s>' #tokenizer.eos_token
        texts.append(text)
    return texts