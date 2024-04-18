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
        dataset = load_dataset("json", data_files="alpaca_Tc_supercon.json", split="train")
    except Exception as e:
        print(e)
        dft_3d = data("dft_3d")
        print(len(dft_3d))
        dataset = make_alpaca_json(dataset=dft_3d, prop="Tc_supercon")
        dumpjson(data=dataset, filename="./alpaca_Tc_supercon.json")
        dataset = Dataset.from_list(dataset)

    # dataset = dataset.map(
    #     eval_prompts,
    #     batched=True,
    # )
    dataset = dataset.add_column("prompt", eval_prompts(dataset))

    df = generate_sample(dataset, model, tokenizer, fourbit_models)

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
    
    