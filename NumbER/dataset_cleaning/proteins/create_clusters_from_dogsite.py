import os
import json
from tqdm import tqdm
import pandas as pd
base_path ="/hpi/fs00/share/fg-naumann/lukas.laskowski/Proteins/output/"
#ids = list(map(lambda x: x.split('.')[0], os.listdir(base_path)))

clusters = os.listdir('/hpi/fs00/share/fg-naumann/lukas.laskowski/Proteins/output/')
result = []
index = 0
gt = []
for cluster in tqdm(clusters):
    gt_cluster = []
    items = os.listdir(os.path.join(base_path, cluster))
    for pdb in items:
        #pdb = pdb.split('.')[0]
        #if pdb == str(id):
        data = pd.read_csv(os.path.join(base_path, cluster, pdb), sep="\t")
        means = data[["lig_cov","poc_cov","lig_name","4A_crit","ligSASRatio","volume","enclosure","surface","lipoSurface",	"depth",	"surf/vol",	"lid/hull",	"ellVol",	"ell_c/a",	"ell_b/a",	"surfGPs",	"lidGPs",	"hullGPs",	"siteAtms",	"accept",	"donor",	"aromat",	"hydrophobicity",	"metal",	"Cs",	"Ns",	"Os",	"Ss",	"Xs",	"acidicAA",	"basicAA",	"polarAA",	"apolarAA",	"sumAA",	"ALA",	"ARG",	"ASN",	"ASP",	"CYS",	"GLN",	"GLU",	"GLY",	"HIS",	"ILE",	"LEU",	"LYS",	"MET",	"PHE",	"PRO",	"SER",	"THR",	"TRP",	"TYR",	"VAL",	"A",	"C",	"G",	"U",	"I",	"N",	"DA",	"DC",	"DG",	"DT",	"DN",	"UNK"]].mean().to_dict()
        means['cluster'] = cluster
        means['name'] = pdb.split("_")[0]
        means['chain'] = pdb.split("_")[1].split(".")[0]
        means['id'] = index
        gt_cluster.append(index)
        result.append(means)
        index += 1
    gt.append(gt_cluster)

pairs = []
for cluster in tqdm(gt):
	for i in range(len(cluster)):
		for j in range(i+1, len(cluster)):
			pairs.append([cluster[i], cluster[j]])
s = pd.DataFrame(pairs, columns=['p1', 'p2'])
s['prediction'] = 1
s.to_csv('all_pairs.csv', index=False)
result = pd.DataFrame(result)
result.to_csv("current_state.csv", index=False)