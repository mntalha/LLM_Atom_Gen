from optimade.client import OptimadeClient
import pandas as pd
import re

# Path to the CSV file containing generated material samples
path = f"./2_gen_mbj_bandgap/material_generated_samples_ternary_all.csv"     
df = pd.read_csv(path)  # Load the CSV into a pandas DataFrame

# Initialize the OptimadeClient (used for querying materials databases)
client = OptimadeClient()

# Dictionary mapping database names to their OPTIMADE API endpoints
databases = {
    "Alexandria": "https://alexandria.icams.rub.de/pbesol",
    "MC3D": "https://aiida.materialscloud.org/mc3d/optimade",
    "OQMD": "https://oqmd.org/optimade/",
    "JARVIS": "https://jarvis.nist.gov/optimade/jarvisdft",
    "MaterialsProject": "https://optimade.materialsproject.org/",
    # "AFLOW": "https://aflow.org/API/optimade/"
}

# Function to swap elements in a chemical formula string
def swap_formula(formula):
    # Extract element symbols and their counts (e.g., ['Na1', 'Cl1'])
    elements = re.findall(r"[A-Z][a-z]?\d*", formula)
    print(elements) # For debugging: print extracted elements
    if len(elements) == 3:
        # For ternary compounds, swap the first and third elements
        elements[0], elements[2] = elements[2], elements[0]
        swapped_formula = "".join(elements)
        return swapped_formula
    if len(elements) == 2: # For binary compounds
        # Swap the two elements
        elements = re.findall(r"[A-Z][a-z]?\d*", formula)
        swapped_formula = "".join(reversed(elements))
        return swapped_formula
    else:
        # If not binary or ternary, return the formula unchanged
        return formula

# Iterate over each row in the DataFrame
for index, row in enumerate(df.iterrows()):
    print("Processing index:", index)
        
    # For each database, try to find the material by its reduced formula
    for database in databases:
        try:
            base_url = databases[database]
            # Re-initialize the client for each database (ensures correct endpoint)
            client = OptimadeClient(base_urls=[base_url])
            # Get the formula from the DataFrame
            formula =  df.loc[index, 'formula']
            # Build the OPTIMADE filter query
            query = f'chemical_formula_reduced="{formula}"'
            # Query the database
            response = client.get(filter=query)

            # If any structures are found, record the count and first structure ID
            if len((response['structures'][query][base_url]['data'])):
                df.loc[index, database] = len((response['structures'][query][base_url]['data']))
                df.loc[index, database +"_id"] = (response['structures'][query][base_url]['data'])[0]['id']
                print(formula, "found..")
                # Save progress after each successful find
                df.to_csv(f"./2_gen_mbj_bandgap/material_generated_samples_database_added.csv", index=False)
            else: 
                # If not found, try swapping the formula elements and search again
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
            # Print error message if something goes wrong
            print("ERRORR Look at", formula, index, database)

# Save the final DataFrame with all results
df.to_csv(f"./2_gen_mbj_bandgap/material_generated_samples_ternary_all_added.csv", index=False)