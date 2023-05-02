import os
import json
from tqdm import tqdm
import pandas as pd
base_path ="/hpi/fs00/share/fg-naumann/lukas.laskowski/Proteins/final_dataset/"
ids = list(map(lambda x: x.split('.')[0], os.listdir(base_path)))
with open("all_clusters.json", "r") as f:
    clusters = json.load(f)
clusters = os.listdir('/hpi/fs00/share/fg-naumann/lukas.laskowski/Proteins/final_dataset/')
result = []

for id in tqdm(ids):
    found = False
    for key in clusters:
        items = os.listdir(os.path.join('/hpi/fs00/share/fg-naumann/lukas.laskowski/Proteins/final_dataset/', key))
        for pdb in items:
            pdb = pdb.split('.')[0]
            if pdb == str(id):
                data = pd.read_csv(os.path.join(base_path, f"{id}.dogsite_desc.txt"), sep="\t")
                means = data[["lig_cov","poc_cov","lig_name","4A_crit","ligSASRatio","volume","enclosure","surface","lipoSurface",	"depth",	"surf/vol",	"lid/hull",	"ellVol",	"ell_c/a",	"ell_b/a",	"surfGPs",	"lidGPs",	"hullGPs",	"siteAtms",	"accept",	"donor",	"aromat",	"hydrophobicity",	"metal",	"Cs",	"Ns",	"Os",	"Ss",	"Xs",	"acidicAA",	"basicAA",	"polarAA",	"apolarAA",	"sumAA",	"ALA",	"ARG",	"ASN",	"ASP",	"CYS",	"GLN",	"GLU",	"GLY",	"HIS",	"ILE",	"LEU",	"LYS",	"MET",	"PHE",	"PRO",	"SER",	"THR",	"TRP",	"TYR",	"VAL",	"A",	"C",	"G",	"U",	"I",	"N",	"DA",	"DC",	"DG",	"DT",	"DN",	"UNK"]].mean().to_dict()
                means['name'] = id
                means['cluster'] = key
                result.append(means)
                found = True
    if not found:
        print(f"DID NOT FIND {id}")
result = pd.concat(result)
result.to_csv("current_state.csv", index=False)