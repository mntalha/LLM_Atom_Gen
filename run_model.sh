#!/bin/sh
#!/bin/sh

# List of model indices to run
model="0 1 2 3 4 5 6 7 8"

# Loop through each model index and run main.py with that index
for mdls in $model;
do
    # Set CUDA device and run the Python script with the current model index
    CUDA_VISIBLE_DEVICES=1 python main.py --model $mdls 
done

# -------------------------------
# Example: How to run this script
# -------------------------------
# 1. Make sure the script is executable:
#    chmod +x run_model.sh
#
# 2. Run the script from the terminal:
#    ./run_model.sh
#
# This will execute main.py for each model index (0 to 8) using CUDA device 1.model="0 1 2 3 4 5 6 7 8"
