import pandas as pd
import numpy as np
from tqdm import tqdm
import os
#"amazon_google", "beer_exp", "company",
datasets = ["amazon_google", "dblp_acm", "dblp_acm_dirty", "dblp_scholar", "dblp_scholar_dirty", "fodors_zagat", "itunes_amazon", "itunes_amazon_dirty", "walmart_amazon", "walmart_amazon_dirty"]
datasets = ["abt_buy", "beer_exp"]

for dataset in tqdm(datasets):
    base_path = f"/hpi/fs00/share/fg-naumann/lukas.laskowski/deep_matcher/{dataset}/"
    df_l_path = f"{base_path}tableA.csv"
    df_r_path = f"{base_path}tableB.csv"
    output_folder = f"/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/single_{dataset}/"

    output_path = f"{output_folder}"
    df_l = pd.read_csv(df_l_path) #l_id
    df_r = pd.read_csv(df_r_path) #r_id
    all_data = pd.concat([df_l, df_r]).drop(columns=['id'])
    all_data['id'] = all_data.index
    #all_data.to_csv(f'{output_path}_all_features.csv', index=False)

    df_l['left_instance_id'] = df_l['id']
    df_r['right_instance_id'] = df_r['id']
    df_r['left_instance_id'] = np.nan
    df_l['right_instance_id'] = np.nan

    df = pd.concat([df_l, df_r], ignore_index=True)
    for gold_standard_old in [[pd.read_csv(f"{base_path}/test.csv"), "test"],[pd.read_csv(f"{base_path}/train.csv"), "train"], [pd.read_csv(f"{base_path}/valid.csv"), "valid"]]:
        sample = gold_standard_old[1]
        gold_standard_old = gold_standard_old[0]
        gold_standard = pd.DataFrame(columns=['left_instance_id', 'right_instance_id', 'label'])
        for idx, row in gold_standard_old.iterrows():
            left_id = df[df['left_instance_id']==row["ltable_id"]].index.values[0]
            right_id = df[df['right_instance_id']==row["rtable_id"]].index.values[0]#.to_csv('./temp.csv')#.idx.values[0]
            status = row['label']
            gold_standard = gold_standard.append({'left_instance_id': left_id, 'right_instance_id': right_id, 'label': status}, ignore_index=True)
        gold_standard = gold_standard.rename(columns={'left_instance_id': 'p1', 'right_instance_id': 'p2', 'label': 'prediction'})
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        gold_standard.to_csv(f'{output_path}{sample}.csv', index=False)
    df.rename(columns={'instance_id': 'id'}, inplace=True)
    df = df.drop(["id", "left_instance_id", "right_instance_id"], axis="columns")
    df['id'] = df.index

    df.to_csv(f'{output_path}features.csv', index=False)
    #gold_standard.to_csv(f'{output_folder}matches.csv', index=False)
    #matches = pd.read_csv(f'{output_folder}matches.csv').rename(columns={'left_instance_id': 'p1', 'right_instance_id': 'p2', 'label': 'prediction'})
    #matches = matches[matches['prediction'] == 1]
    #matches.to_csv(f'{output_folder}matches.csv', index=False)