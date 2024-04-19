from jarvis.db.figshare import data
from datasets import load_dataset
from utils import make_alpaca_json, formatting_prompts_func,eval_prompts
from models import load_model, get_model, get_trainer, save_model
from jarvis.db.jsonutils import loadjson, dumpjson
from unsloth import FastLanguageModel
from datasets import Dataset
from sample_funcs import generate_sample

if __name__ == "__main__":

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

    # from pretrained 
    model, tokenizer = load_model(fourbit_models) #/home/jipengsun/atom-gen/meta-llama
        
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    
    try:
        data_train = load_dataset("json", data_files="./data/alpaca_Tc_supercon_train.json", split="train")
        data_test = load_dataset("json", data_files="./data/alpaca_Tc_supercon_test.json", split="train")
        data_val = load_dataset("json", data_files="./data/alpaca_Tc_supercon_val.json", split="train")
    except Exception as e:
        print(e)

    # dataset = dataset.map(
    #     eval_prompts,
    #     batched=True,
    # )
    data_train = data_train.add_column("prompt", eval_prompts(data_train))

    df = generate_sample(data_train, model, tokenizer, fourbit_models)

    print(df)





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
    
    