from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
import pandas as pd
import random

def parse_fn(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]
    
    structure = Structure(
        lattice=Lattice.from_parameters(
            *(lengths + angles)),
        species=species,
        coords=coords, 
        coords_are_cartesian=False,
    )
    
    return structure.to(fmt="cif")

def ge_samples(data, model, tokenizer, model_name):
    data = data["prompt"][5]
    batch = tokenizer(data, return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items()}
    generate_ids = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True)

    gen_strs = tokenizer.batch_decode(
                    generate_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )
    for gen_str in (gen_strs):
            material_str = gen_str.replace(data, "")
    print(material_str)
def generate_sample(data, model, tokenizer, model_name):

    # count = 0 
    # num_samples = 15 
    # start_range = 0
    # end_range = len(data)
    # random_idx = random.sample(range(start_range, end_range), num_samples * 5)
    prompts = []
    for idx, d_data in enumerate(data):
        prompt = data["prompt"][idx]
        prompts.append(prompt)


    outputs = []
    idx = len(outputs) - 1 
    while len(outputs) < num_samples:
        idx += 1
        batch_prompts = prompts[idx:idx+ 1]

        try:
            batch = tokenizer(list(batch_prompts), return_tensors="pt")
            batch = {k: v.cuda() for k, v in batch.items()}
        except:
            print(idx), batch_prompts
            continue
        generate_ids = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=4096,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True)
        try:
            gen_strs = tokenizer.batch_decode(
                    generate_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )
        except:
            continue
        
        for gen_str, prompt in zip(gen_strs, batch_prompts):
            print(gen_str)
            material_str = gen_str.replace(prompt, "")
            # if material_str.endswith("</s>"):
            #     material_str = material_str[:-4]
            try:
                cif_str = parse_fn(material_str)
                _ = Structure.from_str(cif_str, fmt="cif")
            except Exception as e:
                print(e)
                count += 1
                if count > 2:
                    break
                continue
            count = 0
            print("1 added ...")
            outputs.append({
                    "gen_str": gen_str,
                    "cif": cif_str,
                    "model_name": model_name,
                })
    df = pd.DataFrame(outputs)
    df.to_csv(f"./gen/{model_name.split('/')[1]}_generated_samples.csv", index=False)  

    return df  