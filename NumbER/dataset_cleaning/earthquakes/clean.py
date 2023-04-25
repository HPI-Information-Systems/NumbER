import pandas as pd
import os
def two_sources(df):
    source_1 = "uu"
    source_2 = "us"
    groundtruth = []
    source_1_df = df[df['datasource'] == source_1]
    for idx,row in source_1_df.iterrows():
        source2_row = df[row['event_id'] == df['event_id']]
        source2_row = source2_row[source2_row['datasource'] == source_2]
        print(source2_row)
        if len(source2_row) == 0:
            continue
        #print(all_stars[all_stars['source'] == 'xsc'])
        assert len(source2_row) == 1
        groundtruth.append((source2_row.index[0], idx))
    groundtruth = pd.DataFrame(groundtruth, columns=[source_2, source_1])
    return groundtruth

def multiple_sources(df):
    clusters = df.groupby('event_id').groups
    pairs = []
    for _, cluster in clusters.items():
        if len(cluster)>1:
            #create pairs out of the ids
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    pairs.append((cluster[i], cluster[j]))
    groundtruth = pd.DataFrame(pairs, columns=['p1', 'p2'])
    return groundtruth

                    
earthquakes = pd.read_csv('temp.csv')
#groundtruth = two_sources(earthquakes)
groundtruth = multiple_sources(earthquakes)
groundtruth.to_csv('groundtruth_associated_us.csv', index=False)
earthquakes.to_csv('earthquakes_associated_us.csv')