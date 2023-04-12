import pandas as pd

duplicates = pd.read_excel('./DUPLICATES.xlsx')
clusters = []
unique_ids = set()

for idx, row in duplicates.iterrows():
    if row['Primary record name'] not in unique_ids:
        unique_ids.add(row['Primary record name'])
        unique_ids.add(row['Duplicate names'])
        unique_ids.add(row['Duplicate names.1']) if not pd.isnull(row['Duplicate names.1']) else None
        unique_ids.add(row['Duplicate names.2']) if not pd.isnull(row['Duplicate names.2']) else None
        cluster = set()
        cluster.add(row['Primary record name'])
        cluster.add(row['Duplicate names'])
        cluster.add(row['Duplicate names.1']) if not pd.isnull(row['Duplicate names.1']) else None
        cluster.add(row['Duplicate names.2']) if not pd.isnull(row['Duplicate names.2']) else None
        clusters.append(cluster)
    else:
        for cluster in clusters:
            if row['Primary record name'] in cluster or row['Duplicate names'] in cluster or row['Duplicate names.1'] in cluster or row['Duplicate names.2'] in cluster:
                cluster.add(row['Primary record name'])
                cluster.add(row['Duplicate names'])
                cluster.add(row['Duplicate names.1']) if not pd.isnull(row['Duplicate names.1']) else None
                cluster.add(row['Duplicate names.2']) if not pd.isnull(row['Duplicate names.2']) else None
                unique_ids.add(row['Duplicate names'])
                unique_ids.add(row['Duplicate names.1']) if not pd.isnull(row['Duplicate names.1']) else None
                unique_ids.add(row['Duplicate names.2']) if not pd.isnull(row['Duplicate names.2']) else None
                break
    if idx % 500 == 0:
        print(f"Processed:  {idx}")

length = 0
final_clusters = []
for cluster in clusters:
    length += len(cluster)
    final_clusters.append(list(cluster))
#sort clusters by length
final_clusters.sort(key=len, reverse=True)
print(final_clusters[:20])
print(f"Length: {length/len(clusters)}")
pair_df = pd.DataFrame(columns=['id1', 'id2'])
pairs = []
for cluster in final_clusters:
	for i in range(len(cluster)):
		for j in range(i+1, len(cluster)):
			pairs.append(pd.DataFrame({'id1': cluster[i], 'id2': cluster[j]}, index=[0]))


pair_df = pd.concat(pairs, ignore_index=True)
pair_df.to_csv('pairs.csv', index=False)
