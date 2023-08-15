import pandas as pd

dataset = "2MASS_small"

df = pd.read_csv(f"/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{dataset}/features.csv")
matches = pd.read_csv(f'/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{dataset}/matches_closed.csv')

df['id_new'] = df.index
for match in matches.itertuples():
	matches.at[match.Index, 'p1'] = df[df['id'] == match.p1]['id_new'].values[0]
	matches.at[match.Index, 'p2'] = df[df['id'] == match.p2]['id_new'].values[0]

df.drop(columns=['id'], inplace=True)
df.rename(columns={'id_new': 'id'}, inplace=True)
matches.to_csv(f'/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{dataset}/matches_closed.csv', index=False)
df.to_csv(f'/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{dataset}/features.csv', index=False)