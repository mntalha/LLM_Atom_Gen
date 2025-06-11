import torch, os
import random
import numpy as np
import pandas as pd


def set_seed():
    os.environ["WANDB_ANONYMOUS"] = "must"
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(random_seed)
    except ImportError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
    torch.use_deterministic_algorithms(True)

set_seed()

# Models in Alignn

from pymatgen.analysis.structure_matcher import StructureMatcher
matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
from jarvis.core.atoms import Atoms
from pymatgen.core.structure import Structure
import jarvis 

# Read the files
fourbit_models = [
        "unsloth/tinyllama-chat", #X
        "unsloth/mistral-7b-bnb-4bit", #X
        "unsloth/gemma-7b-bnb-4bit", #X
        "unsloth/llama-3-8b-bnb-4bit", #X 
]  # More models at https://huggingface.co/unsloth

# optb88vdw_bandgap


from models import load_model, get_model, get_trainer, save_model
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd 
from utils import eval_prompts
from sample_funcs import parse_fn
from transformers import TextStreamer
from alignn import pretrained

llm_path = "./0_models_tc_supercon"

for idx, name in enumerate(fourbit_models):

        #Read the model
        fourbit_model = fourbit_models[idx]   
        print("Models Name", fourbit_model)
        llm_model, llm_tokenizer = FastLanguageModel.from_pretrained(os.path.join(llm_path, fourbit_model.split("/")[1])) 
        FastLanguageModel.for_inference(llm_model)  # Enable native 2x faster inference

        data_test = load_dataset("json", data_files="./data/alpaca_"+ "Tc_supercon" +"_test.json", split="train") # #optb88vdw_bandgap  mbj_bandgap Tc_supercon
        data_test = data_test.add_column("prompt", eval_prompts(data_test))
        df = pd.DataFrame(data_test)

        #Look at each models
        for jdx, nname in df.iterrows():
                
                try:
                    batch = llm_tokenizer(df["prompt"][jdx], return_tensors="pt")
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
                    try:
                        material_str = gen_strs[0].replace(df["prompt"][jdx], "")
                        cif_str = parse_fn(material_str)
                        df.loc[jdx, 'gen_material_str'] = material_str
                        df.loc[jdx, 'gen_material_cif'] = cif_str
                        df.loc[jdx, 'orj_material_cif'] = parse_fn(df["response"][jdx])
                    except Exception as e: 
                        print(e)

                    str_pred = Structure.from_str(df["gen_material_cif"][jdx], fmt="cif")
                    str_tar = Structure.from_str(df["orj_material_cif"][jdx], fmt="cif")
                    atoms_pred = jarvis.core.atoms.pmg_to_atoms(str_pred).pymatgen_converter()
                    atoms_tar = jarvis.core.atoms.pmg_to_atoms(str_tar).pymatgen_converter()
                    rms_dist = matcher.get_rms_anonymous(atoms_pred, atoms_tar)
                    df.loc[jdx, 'rms_dist'] = rms_dist[0]
                        
                    try:
                        df.loc[jdx, 'orj_prop_val'] = float(nname["input"].split("value is ")[1][:-1])
                    except Exception as e:
                        print('orj_prop_val')
                        df.loc[jdx, 'orj_prop_val'] = None

                    try:
                        out_data_pred = pretrained.get_prediction(
                                        model_name="jv_supercon_tc_alignn",
                                        atoms=jarvis.core.atoms.pmg_to_atoms(str_pred),
                        )
                        df.loc[jdx, 'out_data_pred'] = out_data_pred[0]
                    except Exception as e:
                        print('out_data_pred')
                        df.loc[jdx, 'out_data_pred'] = None    

                    try:           
                        out_data_tar = pretrained.get_prediction(
                                        model_name="jv_supercon_tc_alignn",
                                        atoms=jarvis.core.atoms.pmg_to_atoms(str_tar),
                        )
                        df.loc[jdx, 'out_data_tar'] = out_data_tar[0]

                    except Exception as e:
                        print('out_data_tar')
                        df.loc[jdx, 'out_data_tar'] = None  
                        
                
                except Exception as e:
                        print(e)
                print("RUNNINNNGGGGG......", jdx)
                

        df.to_csv(f"./0_gen_tc_supercon/{fourbit_model.split('/')[1]}_generated_samples_updated.csv", index=False)
        del(df)  
        
                