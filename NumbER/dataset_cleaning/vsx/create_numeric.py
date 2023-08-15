import pandas as pd

df = pd.read_csv('/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/vsx_small/features.csv')
df = df.drop(columns=['Type', 'n_max', 'n_min'])
matches = pd.read_csv('/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/vsx_small/matches_closed.csv')
df.to_csv('/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/vsx_small_numeric/features.csv', index=False)
matches.to_csv('/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/vsx_small_numeric/matches_closed.csv', index=False)