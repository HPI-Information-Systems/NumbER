import pandas as pd
final_values = []
column = pd.read_csv('../movie1/movies1/csv_files/imdb.csv')['Genre'].unique()
for x in column:
    splits = str(x).split(",")
    for split in splits:
        split = split.replace(" ", "")
        final_values.append(split)
        
print(set(final_values))