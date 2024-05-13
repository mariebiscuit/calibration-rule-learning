import os

subset_dir = "./data/subsets"
SUBSETS = {}

for file in os.listdir(subset_dir):
    with open(os.path.join(subset_dir, file), 'w') as f:
        name = file.split(".")[0]
        SUBSETS[name] = [x.strip() for x in f.readlines()]