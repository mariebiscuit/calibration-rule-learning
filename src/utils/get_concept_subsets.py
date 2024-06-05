import os
import json
"""
Creates the importable SUBSETS dictionary that gives
subsets as described in .txt files in ./data/subsets 
"""
subset_dir = "./data/subsets"
SUBSETS = {}

for file in os.listdir(subset_dir):
    with open(os.path.join(subset_dir, file), 'r') as f:
        name = file.split(".")[0]
        SUBSETS[name] = [x.strip() for x in f.readlines()]

with open("./data/labels_to_readable.json", 'r') as f:
    READABLE = json.load(f)