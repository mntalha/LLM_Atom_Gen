from models import load_model, get_model, get_trainer, save_model
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd 
from utils import eval_prompts
from sample_funcs import parse_fn
from transformers import TextStreamer
import time
import random

fourbit_models = [
        "unsloth/tinyllama-chat", #X
        "unsloth/mistral-7b-bnb-4bit", #X
        "unsloth/gemma-7b-bnb-4bit", #X
        "unsloth/llama-3-8b-bnb-4bit", #X 
]  # More models at https://huggingface.co/unsloth

def path_1(path): 
        
        for model_idx, models in enumerate(fourbit_models):
                
                # LLM Models
                fourbit_model = fourbit_models[model_idx]   
                print("Models Name", fourbit_model)
                llm_model, llm_tokenizer = FastLanguageModel.from_pretrained(os.path.join(path, fourbit_model.split("/")[1])) 
                FastLanguageModel.for_inference(llm_model)  # Enable native 2x faster inference

                # Data Loading
            
                if path == "./0_models_tc_supercon":
                        data_test = load_dataset("json", data_files="./data/alpaca_"+ "Tc_supercon" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap Tc_supercon
                if path == "./1_models_optb88vdw_bandgap":
                        data_test = load_dataset("json", data_files="./data/alpaca_"+ "optb88vdw_bandgap" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap Tc_supercon
                if path == "./2_models_mbj_bandgap":
                        data_test = load_dataset("json", data_files="./data/alpaca_"+ "mbj_bandgap" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap Tc_supercon
                
                data_test = data_test.add_column("prompt", eval_prompts(data_test))


                # LLM OUTPUT GENERATION
                
                start = time.time()

                random_samples = random.sample(data_test["prompt"], 20)
                
                for idx, prompt in enumerate(random_samples):

                # for idx, prompt in enumerate(data_test["prompt"][:20]):

                        batch = llm_tokenizer(prompt, return_tensors="pt")
                        batch = {k: v.cuda() for k, v in batch.items()}
                        generate_ids = llm_model.generate(
                                    **batch,
                                    do_sample=False,
                                    max_new_tokens=4096,
                                    pad_token_id=llm_tokenizer.eos_token_id,
                                    use_cache=True)

                        gen_strs = llm_tokenizer.batch_decode(
                                    generate_ids, 
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True
                                    )
                end = time.time()
                del(llm_model)
                del(llm_tokenizer)
                print('Elapsed time is %f seconds for 20 samples, avearaged %f' % (end-start,(end-start)/20))


print("Randomly Assigned Results")
print("\n\nTC Supercon**************")
path_1("./0_models_tc_supercon")  
print("OPTB88")
path_1("./1_models_optb88vdw_bandgap")   
print("MBJ_BANDGAP")
path_1("./2_models_mbj_bandgap")                         


# Fixed prompt

fourbit_models = [
        "unsloth/tinyllama-chat", #X
        "unsloth/mistral-7b-bnb-4bit", #X
        "unsloth/gemma-7b-bnb-4bit", #X
        "unsloth/llama-3-8b-bnb-4bit", #X 
]  # More models at https://huggingface.co/unsloth

def path_2(path): 
        
        for model_idx, models in enumerate(fourbit_models):
                
                # LLM Models
                fourbit_model = fourbit_models[model_idx]   
                print("Models Name", fourbit_model)
                llm_model, llm_tokenizer = FastLanguageModel.from_pretrained(os.path.join(path, fourbit_model.split("/")[1])) 
                FastLanguageModel.for_inference(llm_model)  # Enable native 2x faster inference

                # Data Loading
            
                if path == "./0_models_tc_supercon":
                        data_test = load_dataset("json", data_files="./data/alpaca_"+ "Tc_supercon" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap Tc_supercon
                if path == "./1_models_optb88vdw_bandgap":
                        data_test = load_dataset("json", data_files="./data/alpaca_"+ "optb88vdw_bandgap" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap Tc_supercon
                if path == "./2_models_mbj_bandgap":
                        data_test = load_dataset("json", data_files="./data/alpaca_"+ "mbj_bandgap" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap Tc_supercon
                
                data_test = data_test.add_column("prompt", eval_prompts(data_test))


                # LLM OUTPUT GENERATION
                
                start = time.time()

                sample = "Below is a description of a superconductor material. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate atomic structure description with lattice lengths, angles, coordinates and atom types.\n\n### Input:\nThe chemical formula is ScZnRh2.\n\n### Response:\n"

                for idx in range(20):

                # for idx, prompt in enumerate(data_test["prompt"][:20]):

                        batch = llm_tokenizer(sample, return_tensors="pt")
                        batch = {k: v.cuda() for k, v in batch.items()}
                        generate_ids = llm_model.generate(
                                    **batch,
                                    do_sample=False,
                                    max_new_tokens=4096,
                                    pad_token_id=llm_tokenizer.eos_token_id,
                                    use_cache=True)

                        gen_strs = llm_tokenizer.batch_decode(
                                    generate_ids, 
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True
                                    )
                end = time.time()
                del(llm_model)
                del(llm_tokenizer)
                print('Elapsed time is %f seconds for 20 samples, avearaged %f' % (end-start,(end-start)/20))

print("Fixed Prompt")
print("\n\nTC Supercon**************")
path_2("./0_models_tc_supercon")  
print("OPTB88")
path_2("./1_models_optb88vdw_bandgap")   
print("MBJ_BANDGAP")
path_2("./2_models_mbj_bandgap")                         



# Nothing

# Fixed prompt

fourbit_models = [
        "unsloth/tinyllama-chat", #X
        "unsloth/mistral-7b-bnb-4bit", #X
        "unsloth/gemma-7b-bnb-4bit", #X
        "unsloth/llama-3-8b-bnb-4bit", #X 
]  # More models at https://huggingface.co/unsloth

def path_3(path): 
        
        for model_idx, models in enumerate(fourbit_models):
                
                # LLM Models
                fourbit_model = fourbit_models[model_idx]   
                print("Models Name", fourbit_model)
                llm_model, llm_tokenizer = FastLanguageModel.from_pretrained(os.path.join(path, fourbit_model.split("/")[1])) 
                FastLanguageModel.for_inference(llm_model)  # Enable native 2x faster inference

                # Data Loading
            
                if path == "./0_models_tc_supercon":
                        data_test = load_dataset("json", data_files="./data/alpaca_"+ "Tc_supercon" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap Tc_supercon
                if path == "./1_models_optb88vdw_bandgap":
                        data_test = load_dataset("json", data_files="./data/alpaca_"+ "optb88vdw_bandgap" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap Tc_supercon
                if path == "./2_models_mbj_bandgap":
                        data_test = load_dataset("json", data_files="./data/alpaca_"+ "mbj_bandgap" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap Tc_supercon
                
                data_test = data_test.add_column("prompt", eval_prompts(data_test))


                # LLM OUTPUT GENERATION
                
                start = time.time()

                sample = ""

                for idx in range(20):

                # for idx, prompt in enumerate(data_test["prompt"][:20]):

                        batch = llm_tokenizer(sample, return_tensors="pt")
                        batch = {k: v.cuda() for k, v in batch.items()}
                        generate_ids = llm_model.generate(
                                    **batch,
                                    do_sample=False,
                                    max_new_tokens=4096,
                                    pad_token_id=llm_tokenizer.eos_token_id,
                                    use_cache=True)

                        gen_strs = llm_tokenizer.batch_decode(
                                    generate_ids, 
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True
                                    )
                end = time.time()
                del(llm_model)
                del(llm_tokenizer)
                print('Elapsed time is %f seconds for 20 samples, avearaged %f' % (end-start,(end-start)/20))

print("Empty Prompt")
print("\n\nTC Supercon**************")
path_3("./0_models_tc_supercon")  
print("OPTB88")
path_3("./1_models_optb88vdw_bandgap")   
print("MBJ_BANDGAP")
path_3("./2_models_mbj_bandgap")                         