import pandas as pd
import os
import json
import subprocess
import sys
from tqdm import tqdm
import swagger_client
import shutil
path_to_dogsite = "~/Masterarbeit/DogSite3/all_nodes\=linux64ubuntu/dogsite_3.0.0/dogsite"
path_to_pdb_files = "/hpi/fs00/share/fg-naumann/lukas.laskowski/Proteins/pdb-files/"
path_to_duplicate_clusters = ""
output_path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/Proteins/output"
dogfile_path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/Proteins/final_dataset"

def main():
    #output_path = "./testus"
    
    with open("../all_clusters.json", "r") as f:
        clusters = json.load(f)
    length = len(clusters.keys())
    result = []#pd.DataFrame(columns=["name","lig_cov","poc_cov","lig_name","4A_crit","ligSASRatio","volume","enclosure","surface","lipoSurface",	"depth",	"surf/vol",	"lid/hull",	"ellVol",	"ell_c/a",	"ell_b/a",	"surfGPs",	"lidGPs",	"hullGPs",	"siteAtms",	"accept",	"donor",	"aromat",	"hydrophobicity",	"metal",	"Cs",	"Ns",	"Os",	"Ss",	"Xs",	"acidicAA",	"basicAA",	"polarAA",	"apolarAA",	"sumAA",	"ALA",	"ARG",	"ASN",	"ASP",	"CYS",	"GLN",	"GLU",	"GLY",	"HIS",	"ILE",	"LEU",	"LYS",	"MET",	"PHE",	"PRO",	"SER",	"THR",	"TRP",	"TYR",	"VAL",	"A",	"C",	"G",	"U",	"I",	"N",	"DA",	"DC",	"DG",	"DT",	"DN",	"UNK"])
    pd.DataFrame(columns=['pdb_id', 'entity', 'chain', 'cluster', 'error']).to_csv('./error_log.csv', sep=";")
    clusters_with_duplicate_pdb_ids = []
    for idx, data in tqdm(enumerate(clusters.items())):
        key = data[0]
        cluster = data[1]
        try:
            pdb_ids = set()
            for el in cluster:
                pdb_id = el[0]
                if pdb_id in pdb_ids:
                    print("PDB ID is already in the set. Handling my dump mistake...")
                    #recalculate the cluster
                    clusters_with_duplicate_pdb_ids.append(key)
                else:
                    pdb_ids.add(pdb_id)
                continue
                pdb_id = pdb_id[0]
                #if "AF" in pdb_id:
                #    raise Exception("AF will not have files. Skipping that cluster.")
                if not os.path.exists(f"{path_to_pdb_files}/{pdb_id}.pdb"):
                    output = os.system(f"wget https://files.rcsb.org/download/{pdb_id}.pdb -P {path_to_pdb_files}")
                    if output != 0:
                        raise Exception("Could not download file, therefore skipping the cluster")
            os.makedirs(f"{output_path}/{key[:-4]}", exist_ok=True)
            cluster_id = key[:-4]
            for pdb_id in cluster:
                try:
                    entity = pdb_id[1]
                    pdb_id = pdb_id[0]
                    if not os.path.exists(f"{path_to_pdb_files}/{pdb_id}.pdb"):
                        raise FileNotFoundError("File does not exist")
                    if cluster_id in clusters_with_duplicate_pdb_ids:
                        #recalculate the cluster
                        calculate_dogfile(cluster_id, pdb_id, entity)
                    elif os.path.exists(f"{dogfile_path}/{cluster_id}/{pdb_id}.dogsite_desc.txt"):
                        shutil.copy(f"{dogfile_path}/{cluster_id}/{pdb_id}.dogsite_desc.txt", f"{output_path}/{cluster_id}/{pdb_id}_{entity}.dogsite_desc.txt")
                    else:
                        calculate_dogfile(cluster_id, pdb_id, entity)
                    #except FileNotFoundError:
                        #continue
                    #os.system(f"{path_to_dogsite} -p {path_to_pdb_files}/{pdb_id}.pdb -d -o {output_path}/{pdb_id}.dogsite -v 1")
                    data = pd.read_csv(f"{output_path}/{key[:-4]}/{pdb_id}_{entity}.dogsite_desc.txt", sep="\t")
                    means = data[["lig_cov","poc_cov","lig_name","4A_crit","ligSASRatio","volume","enclosure","surface","lipoSurface",	"depth",	"surf/vol",	"lid/hull",	"ellVol",	"ell_c/a",	"ell_b/a",	"surfGPs",	"lidGPs",	"hullGPs",	"siteAtms",	"accept",	"donor",	"aromat",	"hydrophobicity",	"metal",	"Cs",	"Ns",	"Os",	"Ss",	"Xs",	"acidicAA",	"basicAA",	"polarAA",	"apolarAA",	"sumAA",	"ALA",	"ARG",	"ASN",	"ASP",	"CYS",	"GLN",	"GLU",	"GLY",	"HIS",	"ILE",	"LEU",	"LYS",	"MET",	"PHE",	"PRO",	"SER",	"THR",	"TRP",	"TYR",	"VAL",	"A",	"C",	"G",	"U",	"I",	"N",	"DA",	"DC",	"DG",	"DT",	"DN",	"UNK"]].mean().to_dict()
                    means['name'] = pdb_id
                    means['entity'] = entity
                    #means['chain'] = chain_id
                    means['cluster'] = key[:-4]
                    result.append(means)
                except Exception as e:
                    print(f"Error {e} occured for {pdb_id}", file=sys.stdout)
                    continue
            print(f"Results calculated and appended {key} with {len(cluster)} elements.", file=sys.stdout)
            if idx % 1000 == 0:
                print(f"Progress: {format(idx/length, '.10f')}")
        except subprocess.TimeoutExpired:
            print(f"Timeout expired for {pdb_id}. Skipping this file.", file=sys.stdout)
        except Exception as e:
            print(f"Did not work for {key} {e}", file=sys.stdout)
            continue
    result = pd.DataFrame(result)
    result.to_csv("/hpi/fs00/share/fg-naumann/lukas.laskowski/Proteins/final_dataset/dogsite.csv", index=False)

def calculate_dogfile(cluster_id, pdb_id, entity):
    timeout_duration = 60 
    apiClient = swagger_client.ApiClient()
    #api_instance = swagger_client.AssemblyServiceApi(swagger_client.ApiClient())
    #holding_service = swagger_client.RepositoryHoldingsServiceApi(apiClient)
    #entry_service = swagger_client.EntryServiceApi(apiClient)
    entity_service = swagger_client.EntityServiceApi(apiClient)
    chain_id = entity_service.get_polymer_entity_by_id(entry_id=pdb_id, entity_id=entity).entity_poly.pdbx_strand_id[0]
    try:
        subprocess.run(f"{path_to_dogsite} -p {path_to_pdb_files}/{pdb_id}.pdb -c {chain_id} -d -o {output_path}/{cluster_id}/{pdb_id}_{entity}.dogsite -v 1", shell=True, timeout=timeout_duration)
    except Exception as e:
        print("Error executing command occured: ", e)
        pd.read_csv('./error_log.csv', sep=";").append({'pdb_id': pdb_id, 'entity': entity, 'chain': chain_id, 'cluster': cluster_id, 'error': e}, ignore_index=True).to_csv('./error_log.csv', sep=";")
    #if not os.path.exists(f"{output_path}/{cluster_id}/{pdb_id}_{entity}.dogsite_desc.txt"):

if __name__ == "__main__":
    main()