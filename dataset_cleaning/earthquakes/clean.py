import pandas as pd
import os
earthquakes = pd.read_csv('earthquakes.csv')
source_1 = "uu"
source_2 = "us"
groundtruth = []
source_1_df = earthquakes[earthquakes['datasource'] == source_1]
for idx,row in source_1_df.iterrows():
    source2_row = earthquakes[row['event_id'] == earthquakes['event_id']]
    source2_row = source2_row[source2_row['datasource'] == source_2]
    print(source2_row)
    if len(source2_row) == 0:
        continue
    #print(all_stars[all_stars['source'] == 'xsc'])
    assert len(source2_row) == 1
    groundtruth.append((source2_row.index[0], idx))
groundtruth = pd.DataFrame(groundtruth, columns=[source_2, source_1])
groundtruth.to_csv('groundtruth.csv', index=False)
