#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:47:34 2024

@author: talha
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = (
    True  # Use 4bit quantization to reduce memory usage. Can be False.
)


def get_model(model_name):
    
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    )
    
    model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    )
    
    model.print_trainable_parameters()
    
    return model, tokenizer

def save_model(model, name):
    
    path = "./model/" + name
    model.save_pretrained(path)
    

def load_model(name):
    
    path = "./model/" + name
    
    model, tokenizer = FastLanguageModel.from_pretrained(path)
    
    return model, tokenizer 
    

def get_trainer(model, tokenizer, data_train, data_val, text, epoch, learning_rate):
    
    trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data_train,
    eval_dataset = data_val, 
    dataset_text_field= text,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        overwrite_output_dir=True,
        # max_steps = 60,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        num_train_epochs=epoch, #5
        report_to="none",
    ),
    )
    
    return trainer