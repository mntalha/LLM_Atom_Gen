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
import torch.nn as nn
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

weighting_ratio = 0.2

def parse(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]
    
    for sp in species:
        lines.remove(sp)
    
    result = []
    for item in lines:
        values = item.split()
        for value in values:
            float_val = float(value)
            result.append(float_val)
    
    result = torch.tensor(result)
    #result.extend((item.split()))
    
    # result.extend(lengths)

    # result.extend(angles)

    # structure = Structure(
    #     lattice=Lattice.from_parameters(
    #         *(lengths + angles)),
    #     species=species,
    #     coords=coords, 
    #     coords_are_cartesian=False,
    # )
    
    return result

class CustomTrainer(SFTTrainer):
    def __init__(self, *args, custom_loss, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = None
        #self.label_smoother = LabelSmoother()
        self.counter = 0
        self.smooth_l1_loss = nn.SmoothL1Loss()


    def compute_loss(self, model, inputs, return_outputs=False):


        outputs = model(**inputs)
        
        loss_smooth = 0
        if self.counter > 3500:
            model.eval()

            generated_ids = inputs.input_ids

            input_text = self.tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True,
                )      

            prompt =  [input_text[0].split("### Response:")[0] + "### Response:\n"]

            batch = self.tokenizer(list(prompt), return_tensors="pt")
            batch = {k: v.cuda() for k, v in batch.items()}
            generate_ids = model.generate(
                    **batch,
                    do_sample=True,
                    max_new_tokens=256,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True)
        
            gen_strs = self.tokenizer.batch_decode(
                    generate_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )

            input_text = self.tokenizer.batch_decode(
                    inputs.labels, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True,
                )
            
            input_ = input_text[0].split("Response:\n")[1]

            try:
                output_ = gen_strs[0].split("Response:\n")[1]
                lines_output = parse(output_)
                lines_input = parse(input_)

                if len(lines_input) > len(lines_output):
                        padding = torch.zeros(len(lines_input) - len(lines_output), dtype=lines_output.dtype)
                        lines_output = torch.cat((lines_output, padding))
                if len(lines_input) < len(lines_output):
                        padding = torch.zeros(len(lines_output) - len(lines_input), dtype=lines_input.dtype)
                        lines_input = torch.cat((lines_input, padding))

                loss_smooth = self.smooth_l1_loss(lines_input, lines_output) * weighting_ratio
                print(loss_smooth)

            except Exception as e:
                print(e)
                loss_smooth = 0 

        
            model.train()

        self.counter += 1
        return outputs.loss + loss_smooth


max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = (
    True  # Use 4bit quantization to reduce memory usage. Can be False.
)

# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()
#         self.loss_fn = nn.CrossEntropyLoss()

#     def forward(self, outputs, labels):
#         logits = outputs.logits
#         B, T, C = logits.shape
#         logits = logits.view(B*T, C)
#         labels = labels.view(B*T)
#         loss = self.loss_fn(logits, labels)
#         # logits = outputs.logits
#         # loss = self.loss_fn(logits, labels)
#         # Add any custom modifications to the loss here
#         return loss

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

def save_model(model, tokenizer, name):
    
    path = "/home/jipengsun/LLM_Atom_Gen/models/" + name #/home/jipengsun/LLM_Atom_Gen/models/llama-3-8b-bnb-4bit
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def load_model(name):
    
    path = "./models/" + name
    
    model, tokenizer = FastLanguageModel.from_pretrained(path)
    
    return model, tokenizer 
    

def get_trainer(model, tokenizer, data_train, data_val, text, epoch, learning_rate):
    

    custom_loss = None #CustomLoss()


    # trainer = CustomTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=data_train,
    #     dataset_text_field= text,
    #     max_seq_length=max_seq_length,
    #     dataset_num_proc=1,
    #     packing=False,  # Can make training 5x faster for short sequences.
    #     args=TrainingArguments(
    #         per_device_train_batch_size=1,
    #         gradient_accumulation_steps=4,
    #         warmup_steps=5,
    #         overwrite_output_dir=True,
    #         #max_steps = None,
    #         learning_rate=learning_rate,
    #         fp16=not torch.cuda.is_bf16_supported(),
    #         bf16=torch.cuda.is_bf16_supported(),
    #         logging_steps=1,
    #         optim="adamw_8bit",
    #         weight_decay=0.01,
    #         lr_scheduler_type="linear",
    #         seed=42,
    #         output_dir="outputs",
    #         num_train_epochs=epoch, #5
    #         report_to="none",
    # ),
    #     custom_loss=custom_loss,
    # )

    trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data_train,
    dataset_text_field= text,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        overwrite_output_dir=True,
        #max_steps = None,
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