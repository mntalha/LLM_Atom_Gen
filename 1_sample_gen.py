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
    parser.add_argument('--model', type=int, default=5,
                         help='0, 1, 2, 3, .. 8') 

    args = parser.parse_args()

    #from pure
    fourbit_models = [
        "unsloth/tinyllama-bnb-4bit", #X
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
        data_train = load_dataset("json", data_files="./data/optb88vdw_bandgap_train.json", split="train")
        data_test = load_dataset("json", data_files="./data/optb88vdw_bandgap_test.json", split="train")
        #data_val = load_dataset("json", data_files="./data/alpaca_Tc_supercon_val.json", split="train")
    except Exception as e:
        print(e)

    # dataset = dataset.map(
    #     eval_prompts,
    #     batched=True,
    # )
    data_train = data_train.add_column("prompt", eval_prompts(data_train))

    #sample_text = "'Below is a description of a superconductor material. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate atomic structure description with lattice lengths, angles, coordinates and atom types.\n\n### Input:\nThe chemical formula is ScZnRh2. The  Tc_supercon value is 0.172. The spacegroup is 225.\n\n### Response:\n4.4 4.4 4.4\n60 60 60\nSc\n0.25 0.25 0.25\nZn\n0.75 0.75 0.75\nRh\n0.00 0.00 0.00\nRh\n0.50 0.50 0.50</s>'"
    #inputs = tokenizer(sample_text, return_tensors='pt')
    #labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    #outputs = model(**inputs, labels=labels)

    # The model's output is a dictionary containing the loss and logits
    #loss = outputs.loss
    #logits = outputs.logits

    #print(f"Loss: {loss.item()}")

    #df = generate_sample(data_train, model, tokenizer, fourbit_models)
    for i in range(4):
        ge_samples(data_train, model, tokenizer, fourbit_models)
    #print(df)





    # input_ch = dataset["prompt"][0]
    # inputs = tokenizer(
    # [
    #    input_ch
    # ],
    # return_tensors="pt",
    # ).to("cuda")

    # outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)

    # output = tokenizer.batch_decode(outputs,skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    # tt = output.replace(input_ch, "")
    #output= tokenizer.batch_decode(outputs)    
    
    