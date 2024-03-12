from NumbER.matching_solutions.utils.transitive_closure import calculate_transitive_closure
import os
import pandas as pd
from tqdm import tqdm

for dataset in tqdm(["dm_fodors_zagat","dm_amazon_google",	"dm_beer_exp",  "dm_dblp_acm",  "dm_dblp_acm_dirty","dm_walmart_amazon",  "dm_walmart_amazon_dirty"]):#os.listdir('/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/'):)
    if os.path.exists(f'/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{dataset}/matches_closed.csv'):#
        print("Already done", dataset)
        continue
    else:
        pairs = calculate_transitive_closure(pd.read_csv(f'/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{dataset}/matches.csv'))
        pairs.to_csv(f'/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{dataset}/matches_closed.csv')