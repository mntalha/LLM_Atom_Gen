#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:08:27 2024

@author: talha
"""
from jarvis.db.figshare import data
from datasets import load_dataset
from models import load_model, get_model, get_trainer, save_model
from jarvis.db.jsonutils import loadjson, dumpjson
from unsloth import FastLanguageModel
from sklearn.model_selection import train_test_split
import json
from utils import make_alpaca_json, formatting_prompts_func
import random
import numpy as np
import torch, os
import os



def set_seed():
    """
    Set random seeds for reproducibility across numpy, random, torch, and torch_xla (if available).
    Also sets deterministic flags for torch and environment variables for reproducibility.
    """
    os.environ["WANDB_ANONYMOUS"] = "must"  # Set Weights & Biases to anonymous mode
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(random_seed)
    except ImportError:
        pass  # torch_xla not available, skip for non-TPU environments
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA deterministic behavior
    torch.use_deterministic_algorithms(True)  # Enforce deterministic algorithms

set_seed()

if __name__ == "__main__":
    

    import argparse
    parser = argparse.ArgumentParser(description='LLM Model Comparison')
    parser.add_argument('--model', type=int, default=1,
                         help='Model index: 0, 1, 2, ... 8') 
    args = parser.parse_args()
                        
    try:
        data_train = load_dataset("json", data_files="./data/alpaca_mbj_bandgap_train.json", split="train")
        data_test = load_dataset("json", data_files="./data/alpaca_mbj_bandgap_test.json", split="train")
        data_val = []
        # data_val = load_dataset("json", data_files="./data/alpaca_Tc_supercon_val.json", split="train")
        print(len(data_train),len(data_test), len(data_val))  
    except Exception as e:
        print(e)
        dft_3d = data("dft_3d")
        print(len(dft_3d))
        
        dataset = make_alpaca_json(dataset=dft_3d, prop="mbj_bandgap") #optb88vdw_bandgap  mbj_bandgap alpaca_Tc_supercon_val
        print(len(dataset))
        train_ratio = 0.90
        test_ratio = 0.10
        #v#al_ratio = 0.05
        data_val = []
        
        # First, split into training and temp sets
        data_train, data_test  = train_test_split(dataset, test_size=(1 - train_ratio), random_state=42)
        
        # relative_val_ratio = val_ratio / (test_ratio + val_ratio)
        # data_test, data_val = train_test_split(data_temp, test_size=relative_val_ratio, random_state=42)
        

        dumpjson(data=data_train, filename="./data/alpaca_mbj_bandgap_train.json")
        dumpjson(data=data_test, filename="./data/alpaca_mbj_bandgap_test.json")
        # dumpjson(data=data_val, filename="./data/alpaca_Tc_supercon_val.json")

        data_train = load_dataset("json", data_files="./data/alpaca_mbj_bandgap_train.json", split="train")
        data_test = load_dataset("json", data_files="./data/alpaca_mbj_bandgap_test.json", split="train")
        # data_val = load_dataset("json", data_files="./data/alpaca_Tc_supercon_val.json", split="train")

    # #from pretrained 
    # #model, tokenizer = load_model()
    
    #from pure
    fourbit_models = [
        "unsloth/tinyllama-chat", #X
        "unsloth/mistral-7b-bnb-4bit", #X
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit", #X
        "unsloth/llama-2-7b-bnb-4bit",  #X
        "unsloth/gemma-7b-bnb-4bit", #X
        "unsloth/llama-3-8b-bnb-4bit", #X 
        "unsloth/llama-2-13b-bnb-4bit", #X
        "unsloth/codellama-34b-bnb-4bit", #X
        "unsloth/llama-3-70b-bnb-4bit",
        "knc6/atomgpt_mistral_tc_supercon",
    ]  # More models at https://huggingface.co/unsloth

    fourbit_models = fourbit_models[args.model]
    print("Models Name", fourbit_models)
    print("Datasets: ",len(data_train), len(data_test))
    model, tokenizer = get_model(fourbit_models)
    #FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    
    data_train = data_train.map(
        lambda batch: formatting_prompts_func(batch, tokenizer),
        batched=True,
    )
    #print(data_train["text"][0])
    data_test = data_test.map(
        lambda batch: formatting_prompts_func(batch, tokenizer),
        batched=True,
    )
    
    trainer = get_trainer(model, tokenizer, data_train, data_val, text="text", epoch=8, learning_rate = 2e-4)

    trainer_stats = trainer.train()
    save_model(model, tokenizer, fourbit_models.split("/")[1])
    dumpjson(data = trainer_stats, filename = "./results/" + fourbit_models.split("/")[1] + "_trainer_stats.json")
    dumpjson(data = trainer.state.log_history, filename = "./results/" + fourbit_models.split("/")[1] + "_trainer_state_log_history.json")

    