#Cleaning code was partly generated with GPT-4
import pandas as pd
import numpy as np
import re
import html
def clean_pattern(x, pattern):
    x = str(x)
    match = re.search(pattern, x)
    if match:
        weight = match.group(1)# or match.group(3)
        weight_value = float(weight)
        return weight_value
    else:
        return ''

def frac_to_decimal(fraction: str) -> float:
    parts = [float(i) for i in fraction.split('/')]
    return parts[0] / parts[1]

# Function to clean width values
def clean_width(value):
    if pd.isna(value):
        return value
    
    value = html.unescape(value)  # Convert HTML entities to their equivalent characters
    value = re.sub(r'[^0-9./-]', '', value)  # Remove double quotes and single quotes
    # Check if there is a range
    if '-' in value:
        range_values = value.split('-')
        try:
            range_start = float(range_values[0])
            range_end = float(range_values[1])
            value = (range_start + range_end) / 2  # Take the average of the range
        except ValueError:
            whole_part, frac_part = value.split('-')
            value = float(whole_part) + frac_to_decimal(frac_part)
    elif '/' in value:
        value = frac_to_decimal(value)  # Convert fractions to decimals
    else:
        if value == '' or value == '.':
            value = np.nan
        value = float(value)
        
    return value


df_1 = pd.read_csv('./baby_products/csv_files/babies_r_us.csv') #r_id
df_2 = pd.read_csv('./baby_products/csv_files/buy_buy_baby.csv') #l_id

df_1['width'] = df_1['width'].str.replace('[^0-9.]', '').replace('"', '').replace('.', np.nan).replace('', np.nan).astype(float)
df_1['height'] = df_1['height'].str.replace('[^0-9.]', '').replace('"', '').replace('.', np.nan).replace('', np.nan).astype(float)
df_1['length'] = df_1['length'].str.replace('[^0-9.]', '').replace('"', '').replace('.', np.nan).replace('', np.nan).astype(float)
df_2['width'] = df_2['width'].apply(clean_width)
df_2['height'] = df_2['height'].apply(clean_width)
df_2['length'] = df_2['length'].apply(clean_width)

colors = ['tan', 'pink', 'nan', 'green', 'espresso', 'gold', 'blue', 'grey', 'mauve', 'orange', 'black', 'cream', 'beige', 'red', 'gray', 'purple', 'chocolate']

oz_pattern = r'(\d+(?:\.\d+)?)\s*oz\b'
lb_pattern = r'(\d+(?:\.\d+)?)\s*(lb|lbs|LBS|pounds|Pounds|Pound|pound)\b'

df_2['weight_lb'] = df_2['weight'].apply(clean_pattern, args=(lb_pattern,))
df_2['weight_oz'] = df_2['weight'].apply(clean_pattern, args=(oz_pattern,))
df_2[['weight', 'weight_lb', 'weight_oz']].to_csv('temp.csv', index=False)

df_1['weight_lb'] = df_1['weight'].apply(clean_pattern, args=(lb_pattern,))
df_1['weight_oz'] = df_1['weight'].apply(clean_pattern, args=(oz_pattern,))
df_1[['weight', 'weight_lb', 'weight_oz']].to_csv('temp.csv', index=False)

df_2['is_discounted']= df_2['is_discounted'].apply(lambda x: 1 if x == True else 0)
df_1['right_instance_id'] = df_1['int_id']
df_2['left_instance_id'] = df_2['int_id']
df_2['right_instance_id'] = np.nan
df_1['left_instance_id'] = np.nan

df = pd.concat([df_1, df_2], ignore_index=True)
gold_standard_old = pd.read_csv('./baby_products/csv_files/labeled_data.csv')
gold_standard = pd.DataFrame(columns=['left_instance_id', 'right_instance_id', 'label'])
for idx, row in gold_standard_old.iterrows():
    left_id = df[df['left_instance_id']==row["ltable.int_id"]].index.values[0]
    right_id = df[df['right_instance_id']==row["rtable.int_id"]].index.values[0]#.to_csv('./temp.csv')#.idx.values[0]
    status = row['product_is_match']
    gold_standard = gold_standard.append({'left_instance_id': left_id, 'right_instance_id': right_id, 'label': status}, ignore_index=True)
gold_standard.to_csv('./baby_products_matches.csv', index=False)
df[['width', 'height', 'length', 'weight_lb', 'weight_oz', 'price', 'is_discounted']].to_csv('./baby_products_features.csv',index_label='instance_id')
    #print("Left id",left_id)
    #print("Right id", right_id)
    #df.loc[df['left_instance_id'] == row.left_instance_id, 'right_instance_id'] = row.right_instance_id
    #df.loc[df['right_instance_id'] == row.right_instance_id, 'left_instance_id'] = row.left_instance_id
#df_1[['width', 'height', 'length', 'weight_lb', 'weight_oz', 'price', 'is_discounted']].to_csv('babies_r_us_only_numbers.csv', index=False)
#df_2[['width', 'height', 'length', 'weight_lb', 'weight_oz', 'price', 'is_discounted']].to_csv('buy_buy_baby_only_numbers.csv', index=False)
