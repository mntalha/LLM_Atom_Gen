
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



if __name__ == "__main__":
    csv_fns = [x for x in glob.glob(f"/home/jipengsun/atom-gen/samples_1.csv") 
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
    
    valid_crys = [x for x in pred_crys if x.valid]
    print("Number of valid crystals: ", len(valid_crys))

    #test data
    gt_cov_cifs = pd.read_csv(args.test_cov_path)["cif"]
    gt_cov_crys_fn = args.test_cov_path.replace(".csv", "_cached.pkl")

    metrics = CDVAEGenEval(
        pred_crys, 
        gt_cov_crys,
        gt_novelty_crys,
        n_samples=len(valid_crys), 
        eval_model_name='mp20'
    ).get_metrics()

    metrics = {k: v for k,v in metrics.items()}
    print(metrics)
    metrics['method'] = args.model_name