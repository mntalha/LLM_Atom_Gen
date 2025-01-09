from alignn import pretrained

models = ["jv_mbj_bandgap_alignn", "jv_optb88vdw_bandgap_alignn", "jv_supercon_tc_alignn"]

align_model = pretrained.get_figshare_model(models[1]) 

from models import load_model, get_model, get_trainer, save_model
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd 
from utils import eval_prompts
from sample_funcs import parse_fn


llm_path = "./1_models_optb88vdw_bandgap"
fourbit_models = [
        "unsloth/tinyllama-chat", #X
        #"unsloth/mistral-7b-bnb-4bit", #X
        "unsloth/gemma-7b-bnb-4bit", #X
        "unsloth/llama-3-8b-bnb-4bit", #X 
]  # More models at https://huggingface.co/unsloth

for model_idx, models in enumerate(fourbit_models):

        # LLM Models

        fourbit_model = fourbit_models[model_idx]   
        print("Models Name", fourbit_model)
        llm_model, llm_tokenizer = FastLanguageModel.from_pretrained(os.path.join(llm_path, fourbit_model.split("/")[1])) 
        FastLanguageModel.for_inference(llm_model)  # Enable native 2x faster inference

        # Data Loading
        
        data_test = load_dataset("json", data_files="./data/alpaca_"+ "optb88vdw_bandgap" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap alpaca_Tc_supercon_val
        data_test = data_test.add_column("prompt", eval_prompts(data_test))
        df = pd.DataFrame(data_test)

        # LLM OUTPUT GENERATION
        
        for idx, prompt in enumerate(data_test["prompt"]):
                print(idx)
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
                material_str = gen_strs[0].replace(prompt, "")
                cif_str = parse_fn(material_str)
                df.loc[idx, 'gen_material_str'] = material_str
                df.loc[idx, 'gen_material_cif'] = cif_str
                df.loc[idx, 'orj_material_cif'] = parse_fn(data_test["response"][idx])

                #df
        df.to_csv(f"./1_gen_optb88vdw_bandgap/{fourbit_model.split('/')[1]}_generated_samples.csv", index=False)
        del(df)  



