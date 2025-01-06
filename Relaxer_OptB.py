import numpy as np
import pandas as pd
from datasets import load_dataset

from jarvis.core.atoms import Atoms
from pymatgen.core.structure import Structure
import jarvis 

from jarvis.db.figshare import get_jid_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jarvis.core.atoms import Atoms, get_supercell_dims
from tqdm import tqdm
from ase.constraints import ExpCellFilter
from sklearn.metrics import mean_absolute_error
import time
from jarvis.core.atoms import ase_to_atoms
from ase.optimize.fire import FIRE
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from alignn.ff.ff import (
    phonons,
    ForceField,
    AlignnAtomwiseCalculator,
    default_path,
)

model_path = default_path()
calc = AlignnAtomwiseCalculator(
        # path=model_path,
        # force_mult_natoms=True,
        # force_multiplier=1,
        # stress_wt=0.3,
    )

# Models in Alignn
from alignn import pretrained

from pymatgen.analysis.structure_matcher import StructureMatcher
matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
from jarvis.core.atoms import Atoms
from pymatgen.core.structure import Structure
import jarvis 

fourbit_models = [
        #"unsloth/tinyllama-chat", #X
        #"unsloth/mistral-7b-bnb-4bit", #X
        "unsloth/gemma-7b-bnb-4bit", #X
        #"unsloth/llama-3-8b-bnb-4bit", #X 
]  # More models at https://huggingface.co/unsloth

# from alignn import pretrained
# pretrained.get_all_models()

import torch, os
import random
import numpy as np
import pandas as pd
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Alternatively, you can directly set the device
#torch.cuda.set_device(0)  # Replace "1" with the index of the GPU you want to use

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

def general_relaxer(atoms="", calculator="", fmax=0.1, steps=250): # fmax=0.1, steps=500 
    ase_atoms = atoms.ase_converter()
    calculator.model.to("cuda")
    ase_atoms.calc = calculator
    ase_atoms = ExpCellFilter(ase_atoms)

    dyn = FIRE(ase_atoms)
    result = dyn.run(fmax=fmax, steps=steps)
    return result, ase_to_atoms(ase_atoms.atoms)


for idx, name in enumerate(fourbit_models):

    # Read the model 
    fourbit_model = fourbit_models[idx]
    path = f"./1_gen_optb88vdw_bandgap/{fourbit_model.split('/')[1]}_generated_samples_updated.csv"
    df = pd.read_csv(path)
    print("********", name, path, len(df))
    
    for jdx, nname in df.iterrows():

        print("RUNNINNNGGGGG......", jdx, name)
        
        # if df.isna().loc[jdx, 'out_data_pred_relaxed'] != True:
        #     continue
        
        print("RUNNINNNGGGGG......", jdx, name)
        try:
            str_pred = Structure.from_str(df["gen_material_cif"][jdx], fmt="cif")
            atoms_pred = jarvis.core.atoms.pmg_to_atoms(str_pred)
            result, opt = general_relaxer(atoms=atoms_pred, calculator=calc)

            if not result:  
                opt = atoms_pred
                
            str_tar = Structure.from_str(df["orj_material_cif"][jdx], fmt="cif")
            atoms_tar = jarvis.core.atoms.pmg_to_atoms(str_tar).pymatgen_converter()
            rms_dist = matcher.get_rms_anonymous(atoms_tar, opt.pymatgen_converter())
            df.loc[jdx, 'rms_dist_relaxed'] = rms_dist[0]

            try:
                out_data_pred = pretrained.get_prediction(
                                model_name="jv_optb88vdw_bandgap_alignn",
                                atoms=opt,
                            )
                df.loc[jdx, 'out_data_pred_relaxed'] = out_data_pred[0]
            except Exception as e:
                print('out_data_pred')
                df.loc[jdx, 'out_data_pred_relaxed'] = None
        except Exception as e:            
            print(e)
            
        df.to_csv(f"./1_gen_optb88vdw_bandgap/{fourbit_model.split('/')[1]}_generated_samples_relaxed.csv", index=False)

        
    df.to_csv(f"./1_gen_optb88vdw_bandgap/{fourbit_model.split('/')[1]}_generated_samples_relaxed.csv", index=False)
    del(df)