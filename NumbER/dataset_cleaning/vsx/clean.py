import pandas as pd

data = pd.read_csv('./final_records_2009.csv')
matches = pd.read_csv('./pairs.csv')
data['id'] = data.index
result = []
for idx, row in matches.iterrows():
    a = row['id1']
    b = row['id2']
    try:
        a = data[data['Name'] == a]['id'].values[0]
        b = data[data['Name'] == b]['id'].values[0]
    except:
        continue
    result.append({'p1': a, 'p2': b, 'prediction': 1})
    
data.drop(columns=['Name', 'OID'], inplace=True)
data.to_csv('./features.csv', index=False)
pd.DataFrame(result).to_csv('./groundtruth.csv', index=False)
    