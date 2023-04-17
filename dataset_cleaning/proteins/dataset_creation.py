import pandas as pd
import os
import json
path_to_dogsite = ""
path_to_pdb_files = ""
path_to_duplicate_clusters = ""
output_path = ""

with open("all_clusters.json", "r") as f:
    clusters = json.load(f)
result = []#pd.DataFrame(columns=["name","lig_cov","poc_cov","lig_name","4A_crit","ligSASRatio","volume","enclosure","surface","lipoSurface",	"depth",	"surf/vol",	"lid/hull",	"ellVol",	"ell_c/a",	"ell_b/a",	"surfGPs",	"lidGPs",	"hullGPs",	"siteAtms",	"accept",	"donor",	"aromat",	"hydrophobicity",	"metal",	"Cs",	"Ns",	"Os",	"Ss",	"Xs",	"acidicAA",	"basicAA",	"polarAA",	"apolarAA",	"sumAA",	"ALA",	"ARG",	"ASN",	"ASP",	"CYS",	"GLN",	"GLU",	"GLY",	"HIS",	"ILE",	"LEU",	"LYS",	"MET",	"PHE",	"PRO",	"SER",	"THR",	"TRP",	"TYR",	"VAL",	"A",	"C",	"G",	"U",	"I",	"N",	"DA",	"DC",	"DG",	"DT",	"DN",	"UNK"])
for key, cluster in clusters.items():
    for pdb_id in cluster:
        if not os.path.exists(f"{path_to_pdb_files}/{pdb_id}.pdb"):
            os.system(f"wget https://files.rcsb.org/download/{pdb_id}.pdb -P {path_to_pdb_files}")
    for pdb_id in cluster:
        os.system(f"{path_to_dogsite} -p {path_to_pdb_files}/{pdb_id}.pdb -d -o {output_path}/{pdb_id}.dogsite")
        data = pd.read_csv(f"{output_path}/{pdb_id}.dogsite_desc.txt", sep="\t")
        means = data[["lig_cov","poc_cov","lig_name","4A_crit","ligSASRatio","volume","enclosure","surface","lipoSurface",	"depth",	"surf/vol",	"lid/hull",	"ellVol",	"ell_c/a",	"ell_b/a",	"surfGPs",	"lidGPs",	"hullGPs",	"siteAtms",	"accept",	"donor",	"aromat",	"hydrophobicity",	"metal",	"Cs",	"Ns",	"Os",	"Ss",	"Xs",	"acidicAA",	"basicAA",	"polarAA",	"apolarAA",	"sumAA",	"ALA",	"ARG",	"ASN",	"ASP",	"CYS",	"GLN",	"GLU",	"GLY",	"HIS",	"ILE",	"LEU",	"LYS",	"MET",	"PHE",	"PRO",	"SER",	"THR",	"TRP",	"TYR",	"VAL",	"A",	"C",	"G",	"U",	"I",	"N",	"DA",	"DC",	"DG",	"DT",	"DN",	"UNK"]].mean().to_dict()
        means['name'] = pdb_id
        means['cluster'] = key
        result.append(means)
result = pd.DataFrame(result)
result.to_csv("dogsite.csv", index=False)
