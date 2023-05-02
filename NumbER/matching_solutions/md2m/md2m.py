import pandas as pd
from NumbER.matching_solutions.md2m.column import Column
from NumbER.matching_solutions.md2m.table import Table

df = pd.read_csv("~/Masterarbeit/NumbER/NumbER/dataset_cleaning/earthquakes/earthquakes_associated_us.csv")
columns = []
for col in df:
    try:
        df[col].astype(float)
    except Exception as e:
        print(e)
        continue
    columns.append(Column(col, df[col]))
table = Table("Table1", columns, df)
table.calculate_candidates()