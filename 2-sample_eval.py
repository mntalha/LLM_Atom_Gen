
from jarvis.db.figshare import data
from datasets import load_dataset
from utils import make_alpaca_json, formatting_prompts_func,eval_prompts
from models import load_model, get_model, get_trainer, save_model
from jarvis.db.jsonutils import loadjson, dumpjson
from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd 
from eval_functions import *
import glob
from p_tqdm import p_map
from sample_funcs import parse_fn


if __name__ == "__main__":

    try:
        data_tr = load_dataset("json", data_files="./data/alpaca_Tc_supercon_train.json", split="train")
        data_te = load_dataset("json", data_files="./data/alpaca_Tc_supercon_test.json", split="train")
    except Exception as e:
        print(e)
     
    data_train = []
    for i in (data_tr):
         ciff = parse_fn(i["output"])
         data_train.append(ciff)

    data_test = []
    for i in (data_te):
         ciff = parse_fn(i["output"])
         data_test.append(ciff)

    fourbit_models = [
        "unsloth/mistral-7b-bnb-4bit", #X
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit", #X
        "unsloth/llama-2-7b-bnb-4bit",  #X
        "unsloth/llama-2-13b-bnb-4bit", #X
        "unsloth/codellama-34b-bnb-4bit", #X
        "unsloth/tinyllama-bnb-4bit", #X
        "meta-llama/Llama-2-7b-hf", 
        "unsloth/llama-3-8b-bnb-4bit",  #X
        "unsloth/llama-3-70b-bnb-4bit",
    ]  # More models at https://huggingface.co/unsloth

    fourbit_models = fourbit_models[7]

    csv_fns = [x for x in glob.glob(f"{fourbit_models.split('/')[1]}_generated_samples.csv") 
            if len(open(x).readlines()) > 1 and 'm3gnet_relaxed_energy' not in x
    ]
    # print(csv_fns)
    pred_cifs = []
    for x in csv_fns:
            try:
                df = pd.read_csv(x)
                pred_cifs += list(df["cif"].dropna())
            except Exception as e:
                print(e)
    pred_cifs = pred_cifs[::-1]
    print(len(pred_cifs))

    pred_crys = [x for x in p_map(cif_str_to_crystal, pred_cifs) if x is not None]

    if len(pred_crys) > 10000:
        random_idx = np.random.choice(len(pred_crys), 10000)
        pred_crys = [pred_crys[x] for x in random_idx]
    
    #test data test
    gt_cov_cifs = data_test #pd.read_csv(args.test_cov_path)["cif"]
    #gt_cov_crys_fn = args.test_cov_path.replace(".csv", "_cached.pkl")
    gt_cov_crys = p_map(cif_str_to_crystal, gt_cov_cifs)

    gt_novelty_cifs = data_train
    gt_novelty_crys = p_map(cif_str_to_crystal, gt_novelty_cifs)    

    valid_crys = [x for x in pred_crys if x.valid]
    print("Number of valid crystals: ", len(valid_crys))

    metrics = CDVAEGenEval(
        pred_crys, 
        gt_cov_crys,
        gt_novelty_crys,
        n_samples=len(valid_crys), 
        eval_model_name='mp20' #check if it true 
    ).get_metrics()

    metrics = {k: v for k,v in metrics.items()}
    print(metrics)
