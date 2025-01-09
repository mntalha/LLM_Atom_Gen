from jarvis.db.figshare import data
from datasets import load_dataset
from utils import make_alpaca_json, formatting_prompts_func,eval_prompts
from models import load_model, get_model, get_trainer, save_model
from jarvis.db.jsonutils import loadjson, dumpjson
from unsloth import FastLanguageModel
from datasets import Dataset
from sample_funcs import generate_sample, ge_samples
import torch
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='LLM Model Comparison')
    parser.add_argument('--model', type=int, default=0,
                         help='0, 1, 2, 3, .. 8') 

    args = parser.parse_args()

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
    ]  # More models at https://huggingface.co/unsloth
    
    # fourbit_models = fourbit_models[args.model]
    fourbit_models = fourbit_models[args.model]   
    print("Models Name", fourbit_models)
    # from pretrained 
    model, tokenizer = load_model(fourbit_models.split("/")[1]) #/home/jipengsun/atom-gen/meta-llama
        
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    
    try:
        data_train = load_dataset("json", data_files="./data/alpaca_Tc_supercon_train.json", split="train")
        data_test = load_dataset("json", data_files="./data/alpaca_Tc_supercon_test.json", split="train")
        #data_val = load_dataset("json", data_files="./data/alpaca_Tc_supercon_val.json", split="train")
    except Exception as e:
        print(e)

    # dataset = dataset.map(
    #     eval_prompts,
    #     batched=True,
    # )
    data_test = data_test.add_column("prompt", eval_prompts(data_test))

    df = generate_sample(data_test, model, tokenizer, fourbit_models)
    print(df)
  