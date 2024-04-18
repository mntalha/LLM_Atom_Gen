from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
import pandas as pd


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

def generate_sample(data, model, tokenizer, model_name):

    prompts = []
    for i in range(30):
        prompt = data["prompt"][i]
        prompts.append(prompt)


    outputs = []
    while len(outputs) < 30:
        batch_prompts = prompts[len(outputs):len(outputs)+ 1]

        batch = tokenizer(list(batch_prompts), return_tensors="pt",)
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
                **batch,
                do_sample=True,
                max_new_tokens=512,
                use_cache=True)
        
        gen_strs = tokenizer.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
        
        for gen_str, prompt in zip(gen_strs, batch_prompts):

            material_str = gen_str.replace(prompt, "")
            try:
                cif_str = parse_fn(material_str)
                _ = Structure.from_str(cif_str, fmt="cif")
            except Exception as e:
                print(e)
                continue
            outputs.append({
                    "gen_str": gen_str,
                    "cif": cif_str,
                    "model_name": model_name,
                })
    df = pd.DataFrame(outputs)
    df.to_csv(f"generated_samples.csv", index=False)  

    return df  