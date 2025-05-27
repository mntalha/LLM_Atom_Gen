from optimade.client import OptimadeClient
import pandas as pd
import re

path = f"./2_gen_mbj_bandgap/material_generated_samples_ternary_all.csv"     
df = pd.read_csv(path)

# Initialize the client
client = OptimadeClient()

databases = {
    "Alexandria": "https://alexandria.icams.rub.de/pbesol",
    "MC3D": "https://aiida.materialscloud.org/mc3d/optimade",
    "OQMD": "https://oqmd.org/optimade/",
    "JARVIS": "https://jarvis.nist.gov/optimade/jarvisdft",
    "MaterialsProject": "https://optimade.materialsproject.org/",
    # "AFLOW": "https://aflow.org/API/optimade/"
}

# Function to swap elements in the formula
def swap_formula(formula):
    # Extract element symbols and their counts
    elements = re.findall(r"[A-Z][a-z]?\d*", formula)
    print(elements) #ternary
    if len(elements) == 3:
        # Swap first and third elements
        elements[0], elements[2] = elements[2], elements[0]
        
        swapped_formula = "".join(elements)
        return swapped_formula
    if len(elements) == 2: #binary
        # Swap first and second elements
        elements = re.findall(r"[A-Z][a-z]?\d*", formula)  # Extract element-symbol + count
        swapped_formula = "".join(reversed(elements))  # Reverse order
        return swapped_formula
    else:
        return formula




for index, row in enumerate(df.iterrows()):

    print("Processing index:", index)
        
    for database in databases:
        # Initialize the client for JARVIS
        try:
            base_url = databases[database]
            client = OptimadeClient(base_urls=[base_url])
            # Define the filter query for a specific reduced formula
            formula =  df.loc[index, 'formula']
            query = f'chemical_formula_reduced="{formula}"'
            response = client.get(filter=query)

            if len((response['structures'][query][base_url]['data'])):
                df.loc[index, database] = len((response['structures'][query][base_url]['data']))
                df.loc[index, database +"_id"] = (response['structures'][query][base_url]['data'])[0]['id']
                print(formula, "found..")
                df.to_csv(f"./2_gen_mbj_bandgap/material_generated_samples_database_added.csv", index=False)
            else: 
                formula = swap_formula(formula)
                query = f'chemical_formula_reduced="{formula}"'
                response = client.get(filter=query)
                if len((response['structures'][query][base_url]['data'])):
                    df.loc[index, database] = len((response['structures'][query][base_url]['data']))
                    df.loc[index, database +"_id"] = (response['structures'][query][base_url]['data'])[0]['id']
                    print("swapped found..", formula)
                    df.to_csv(f"./2_gen_mbj_bandgap/material_generated_samples_database_added.csv", index=False)
                else:
                    print("Not Found..")
        except Exception as e: 
            print("ERRORR Look at", formula, index, database)

df.to_csv(f"./2_gen_mbj_bandgap/material_generated_samples_ternary_all_added.csv", index=False)