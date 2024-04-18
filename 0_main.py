#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:08:27 2024

@author: talha
"""
from jarvis.db.figshare import data
from datasets import load_dataset
from utils import make_alpaca_json, formatting_prompts_func
from models import load_model, get_model, get_trainer, save_model
from jarvis.db.jsonutils import loadjson, dumpjson
from unsloth import FastLanguageModel
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    try:
        data_train = load_dataset("json", data_files="./data/alpaca_Tc_supercon_train.json")
        data_test = load_dataset("json", data_files="./data/alpaca_Tc_supercon_test.json")
        data_val = load_dataset("json", data_files="./data/alpaca_Tc_supercon_val.json")
    except Exception as e:
        print(e)
        dft_3d = data("dft_3d")
        print(len(dft_3d))
        
        dataset = make_alpaca_json(dataset=dft_3d, prop="Tc_supercon")

        train_ratio = 0.85
        test_ratio = 0.10
        val_ratio = 0.05
        # First, split into training and temp sets
        data_train, data_temp  = train_test_split(dataset, test_size=(1 - train_ratio), random_state=42)
        
        relative_val_ratio = val_ratio / (test_ratio + val_ratio)
        data_test, data_val = train_test_split(data_temp, test_size=relative_val_ratio, random_state=42)
        
        dumpjson(data=data_train, filename="./data/alpaca_Tc_supercon_train.json")
        dumpjson(data=data_test, filename="./data/alpaca_Tc_supercon_test.json")
        dumpjson(data=data_val, filename="./data/alpaca_Tc_supercon_val.json")

        data_train = load_dataset("json", data_files="./data/alpaca_Tc_supercon_train.json")
        data_test = load_dataset("json", data_files="./data/alpaca_Tc_supercon_test.json")
        data_val = load_dataset("json", data_files="./data/alpaca_Tc_supercon_val.json")


    
    # #from pretrained 
    # #model, tokenizer = load_model()
    
    #from pure
    fourbit_models = [
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/llama-2-7b-bnb-4bit",
        "unsloth/llama-2-13b-bnb-4bit",
        "unsloth/codellama-34b-bnb-4bit",
        "unsloth/tinyllama-bnb-4bit",
        "meta-llama/Llama-2-7b-hf"
    ]  # More models at https://huggingface.co/unsloth

    fourbit_models = fourbit_models[6]
    model, tokenizer = get_model(fourbit_models)
    
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    
    trainer = get_trainer(model, tokenizer, data_train, data_val, text="text", epoch= 1, learning_rate = 3e-4)

    trainer_stats = trainer.train()
    save_model(model, fourbit_models)
    dumpjson(data = trainer_stats, filename = "./results/" + fourbit_models + "_trainer_stats.json")
    dumpjson(data = trainer.state.log_history, filename = "./results/" + fourbit_models + "_trainer_state_log_history.json")

    