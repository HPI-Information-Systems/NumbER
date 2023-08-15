import pandas as pd
from sklearn import preprocessing

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

le = preprocessing.LabelEncoder()
le.fit([*data['n_max']])
data['n_max_categorical'] = le.transform(data['n_max'])
le = preprocessing.LabelEncoder()
le.fit([*data['n_min']])
data['n_min_categorical'] = le.transform(data['n_min'])
data.drop(columns=['OID'], inplace=True)
data.to_csv('./features_combined.csv', index=False)
data.drop(columns=['Name', 'Type', 'n_max', 'n_min', 'f_min', 'l_min'], inplace=True)
data.to_csv('./features.csv', index=False)
pd.DataFrame(result).to_csv('./groundtruth.csv', index=False)
#also removed in _small the attributes: 'u_max', 'f_min', 'l_min', 'u_min', 'u_Epoch', 'l_Period', 'u_Period'