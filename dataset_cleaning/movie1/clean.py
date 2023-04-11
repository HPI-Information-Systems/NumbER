import pandas as pd
import re
from sklearn import preprocessing
import numpy as np

def duration_to_minutes(duration_str):
    if pd.isna(duration_str):
        return None
    else:
        duration_parts = duration_str.split()
        hours = int(duration_parts[0]) if 'hr.' in duration_str else 0
        minutes = int(duration_parts[-2]) if 'min.' in duration_str else 0
        return 60 * hours + minutes
    
def process_rating(rating_str):
    if pd.isna(rating_str):
        return None
    else:
        ratings = re.findall(r'(\d+(\.\d+)?)/10', rating_str)
        ratings = [float(rating[0]) for rating in ratings]
        return sum(ratings) / len(ratings)

df_1 = pd.read_csv('./movies1/csv_files/imdb.csv') #r_id
df_2 = pd.read_csv('./movies1/csv_files/rotten_tomatoes.csv')#l_id

df_1['Duration'] = df_1['Duration'].apply(duration_to_minutes).astype(float)
df_2['Duration'].str.replace(' min', '').astype(float)
print(df_1['ReleaseDate'].str.extract('([A-Za-z]{3} \d{1,2}, \d{4})').to_csv('./temp.csv'))
print(df_2['Release Date'].str.extract('(\d{1,2} \w+ \d{4})').to_csv('./temp2.csv'))
#df_1['ReleaseDate'].fillna('', inplace=True)
df_1['ReleaseDate'] = pd.to_datetime(df_1['ReleaseDate'].str.extract('(\w{3} \d{1,2}, \d{4})').iloc[:, 0],errors='ignore', format='%b %d, %Y')#.astype(float)
df_2['ReleaseDate'] = pd.to_datetime(df_2['Release Date'].str.extract('(\d{1,2} \w+ \d{4})').iloc[:, 0],errors='ignore', format='%d %B %Y')#.astype(float)
df_1['RatingValue'] = df_1['RatingValue'].apply(process_rating)

le = preprocessing.LabelEncoder()
le.fit([*df_1['ContentRating']])#,*df_2['Publisher']])
df_1['ContentRating'] = le.transform(df_1['ContentRating'])
le.fit([*df_1['Genre']])#,*df_2['Publisher']])
df_1['Genre'] = le.transform(df_1['Genre']) #Gedanken machen, wie man mit genre umgeht. Eine Spalte pro Genre?

df_1 = df_1[['Id', 'RatingValue', 'Duration', 'ReleaseDate']]
df_2 = df_2[['Id', 'RatingValue', 'Duration', 'ReleaseDate']]

df_1['right_instance_id'] = df_1['Id']
df_2['left_instance_id'] = df_2['Id']
df_2['right_instance_id'] = np.nan
df_1['left_instance_id'] = np.nan

df = pd.concat([df_1, df_2], ignore_index=True)
gold_standard_old = pd.read_csv('./movies1/csv_files/labeled_data.csv')
gold_standard = pd.DataFrame(columns=['left_instance_id', 'right_instance_id', 'label'])
for idx, row in gold_standard_old.iterrows():
    left_id = df[df['left_instance_id']==row["ltable.Id"]].index.values[0]
    right_id = df[df['right_instance_id']==row["rtable.Id"]].index.values[0]#.to_csv('./temp.csv')#.idx.values[0]
    status = row['gold']
    gold_standard = gold_standard.append({'left_instance_id': left_id, 'right_instance_id': right_id, 'label': status}, ignore_index=True)

df.drop(["Id", "left_instance_id", "right_instance_id"], axis="columns").to_csv('./movies1_features.csv',index_label='instance_id')
gold_standard.to_csv('./movies1_matches.csv', index=False)