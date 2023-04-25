import pandas as pd
import numpy as np
from sklearn import preprocessing

def extract_dimensions_half(dimensions_str):
    dimensions_str = str(dimensions_str)
    try:
        splits = dimensions_str.split("x")
        height = splits[0].replace(" ", "")
        width = splits[1].replace(" ", "")
        length = splits[2].split(" ")[1]  
        return height, width, length 
    except:
        return None, None, None
def extract_dimensions_barnes(dimensions_str):
    dimensions_str = str(dimensions_str)
    try:
        splits = dimensions_str.split("x")
        width = splits[0].replace(" ", "")[:-3]
        height = splits[1].replace(" ", "")[:-3]
        length = splits[2].replace(" ", "")[:-3]
        return height, width, length 
    except:
        return None, None, None

df_1 = pd.read_csv('./books3/csv_files/barnes_and_noble.csv') #l_id
df_2 = pd.read_csv('./books3/csv_files/half.csv') #r_id

df_1['Price'] = df_1['Price'].str.replace("$", "").astype(float)
df_2['UsedPrice'] = df_2['UsedPrice'].str.replace("$", "").astype(float)
df_2['NewPrice'] = df_2['NewPrice'].str.replace("$", "").astype(float)

# Create new columns for height, width, and length
df_1['height'], df_1['width'], df_1['length'] = zip(*df_1['Dimensions'].apply(extract_dimensions_barnes))
df_2['height'], df_2['width'], df_2['length'] = zip(*df_2['Dimensions'].apply(extract_dimensions_half))

le = preprocessing.LabelEncoder()
le.fit([*df_1['Publisher'],*df_2['Publisher']])
df_1['Publisher'] = le.transform(df_1['Publisher'])
df_2['Publisher'] = le.transform(df_2['Publisher'])

df_1 = df_1[['ID', 'Price', 'Pages','ISBN13', 'Publisher', 'height', 'width', 'length']]
#df_2 = df_2[['ID', 'UsedPrice', 'NewPrice', 'ISBN13', 'Publisher', 'height', 'width', 'length']]
df_2['Price'] = df_2['UsedPrice']
df_2 = df_2[['ID', 'Price', 'Pages','ISBN13', 'Publisher', 'height', 'width', 'length']]

df_1['left_instance_id'] = df_1['ID']
df_2['right_instance_id'] = df_2['ID']
df_2['left_instance_id'] = np.nan
df_1['right_instance_id'] = np.nan

df = pd.concat([df_1, df_2], ignore_index=True)
gold_standard_old = pd.read_csv('./books3/csv_files/labeled_data.csv')
gold_standard = pd.DataFrame(columns=['left_instance_id', 'right_instance_id', 'label'])
for idx, row in gold_standard_old.iterrows():
    left_id = df[df['left_instance_id']==row["ltable.ID"]].index.values[0]
    right_id = df[df['right_instance_id']==row["rtable.ID"]].index.values[0]#.to_csv('./temp.csv')#.idx.values[0]
    status = row['gold']
    gold_standard = gold_standard.append({'left_instance_id': left_id, 'right_instance_id': right_id, 'label': status}, ignore_index=True)

df.drop(["ID", "left_instance_id", "right_instance_id"], axis="columns").to_csv('./books3_features.csv',index_label='instance_id')
gold_standard.to_csv('./books3_matches.csv', index=False)


#df_1[['ID', 'Price', 'Pages','ISBN13', 'Publisher', 'height', 'width', 'length']].to_csv('./barnes_and_noble_numeric.csv', index=False)
#df_1[['ID', 'Price', 'Pages','ISBN13', 'height', 'width', 'length']].to_csv('./barnes_and_noble_numeric_without_categoric.csv', index=False)

#df_2[['ID', 'UsedPrice', 'NewPrice', 'Pages','ISBN13', 'Publisher', 'height', 'width', 'length']].to_csv('./half_numeric.csv', index=False)
#df_2[['ID', 'UsedPrice', 'NewPrice', 'Pages', 'ISBN10', 'ISBN13', 'height', 'width', 'length']].to_csv('./half_numeric_without_categoric.csv', index=False)