import pandas as pd
import numpy as np
from sklearn import preprocessing
import re

def clean_pattern(x, pattern):
    x = str(x)
    match = re.search(pattern, x)
    if match:
        weight = match.group(1)# or match.group(3)
        weight_value = float(weight)
        return weight_value
    else:
        return ''

def extract_dimensions_amazon(dimensions_str):
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
    
def extract_prices(price_data):
    if pd.isnull(price_data):
        return None, None
    price_data = re.sub(r'\s*Save.*', '', price_data)
    price_data = re.sub(r'Pre-order Price:\s*', '', price_data)
    prices = re.findall(r'\$([\d,]+(\.\d{2})?)', price_data)
    prices = [float(price[0].replace(',', '')) for price in prices]
    if len(prices) == 1:
        min_price, max_price = prices[0], prices[0]
        return min_price
    else:
        min_price, max_price = min(prices), max(prices)
        return max_price
    #return min_price, max_price

df_1 = pd.read_csv('./books4/csv_files/amazon.csv') #l_id
df_2 = pd.read_csv('./books4/csv_files/barnes_and_noble.csv') #r_id

#df_1['Price'] = df_1['Price'].str.replace("$", "").replace("\xa0",np.nan).astype(float)

#df_1['min_price'], df_1['max_price'] = zip(*df_1['Price'].apply(extract_prices))
df_1['Price'] = df_1['Price'].apply(extract_prices)
# df_1[['min_price', 'max_price']].to_csv('./temp.csv', index=False)

df_2['Hardcover'] = df_2['Hardcover'].str.replace("$", "").replace("\xa0",np.nan).astype(float)
df_2['Paperback'] = df_2['Paperback'].str.replace("$", "").replace("\xa0",np.nan).astype(float)
df_2['NOOK_Book'] = df_2['NOOK_Book'].str.replace("$", "").replace("\xa0",np.nan).astype(float)
df_2['Audiobook'] = df_2['Audiobook'].str.replace("$", "").replace("\xa0",np.nan).astype(float)
df_2['ISBN_13'] = df_2['ISBN_13'].str.replace(r'[^0-9]', "").replace("",np.nan).astype(float)
df_1['ISBN_13'] = df_1['ISBN_13'].str.replace(r'[^0-9]', "").replace("",np.nan).astype(float)
df_1['ISBN_10'] = df_1['ISBN_10'].str.replace(r'[^0-9]', "").replace("",np.nan).astype(float)
df_1['Paperback'].str.replace(r"(pages| )", "").astype(float)
df_2['Sales_rank'] = df_2['Sales_rank'].str.replace(",","").astype(float)

#Convert shipping weight
lb_pattern = r'(\d+(?:\.\d+)?)\s*(lb|lbs|LBS|pounds|Pounds|Pound|pound)\b'
oz_pattern = r'(\d+(?:\.\d+)?)\s*oz\b'
df_1['weight_lb'] = df_1['Shipping Weight'].apply(clean_pattern, args=(lb_pattern,))
df_1['weight_oz'] = df_1['Shipping Weight'].apply(clean_pattern, args=(oz_pattern,))

# Create new columns for height, width, and length
df_1['height'], df_1['width'], df_1['length'] = zip(*df_1['Product Dimensions'].apply(extract_dimensions_amazon))
df_2['height'], df_2['width'], df_2['length'] = zip(*df_2['Product_dimensions'].apply(extract_dimensions_barnes))

le = preprocessing.LabelEncoder()
#df_1['Publisher'] = le.fit_transform(df_1['Publisher'])
# df_1['Edition'] = le.fit_transform(df_1['Edition'])
# df_2['Publisher'] = le.fit_transform(df_2['Publisher'])
# df_2['Sold_by'] = le.fit_transform(df_2['Sold_by'])
# df_2['Language'] = le.fit_transform(df_2['Language'])

le.fit([*df_1['Publisher'].unique(), *df_2['Publisher'].unique()])
df_2["Publisher"] = le.transform(df_2["Publisher"])
df_1["Publisher"] = le.transform(df_1["Publisher"])

df_2["Price"] = df_2["Paperback"].fillna(df_2["Hardcover"])

df_1 = df_1[["ID", "Publication_Date", "Publisher", "Price", "ISBN_13", "height", "width", "length"]]
df_2 = df_2[["ID", "Publication_Date", "Publisher", "Price", "ISBN_13", "height", "width", "length"]]
df_1['left_instance_id'] = df_1['ID']
df_2['right_instance_id'] = df_2['ID']
df_2['left_instance_id'] = np.nan
df_1['right_instance_id'] = np.nan

df = pd.concat([df_1, df_2], ignore_index=True)
gold_standard_old = pd.read_csv('./books4/csv_files/labeled_data.csv')
gold_standard = pd.DataFrame(columns=['left_instance_id', 'right_instance_id', 'label'])
for idx, row in gold_standard_old.iterrows():
    left_id = df[df['left_instance_id']==row["ltable.ID"]].index.values[0]
    right_id = df[df['right_instance_id']==row["rtable.ID"]].index.values[0]#.to_csv('./temp.csv')#.idx.values[0]
    status = row['gold']
    gold_standard = gold_standard.append({'left_instance_id': left_id, 'right_instance_id': right_id, 'label': status}, ignore_index=True)

df.drop(["ID", "left_instance_id", "right_instance_id"], axis="columns").to_csv('./books4_features.csv',index_label='instance_id')
gold_standard.to_csv('./books4_matches.csv', index=False)
# df_1[['ID', 'Price', 'Pages', 'ISBN13', 'Publisher', 'height', 'width', 'length']].to_csv('./barnes_and_noble_numeric.csv', index=False)
# df_1[['ID', 'Price', 'Pages', 'ISBN13', 'height', 'width', 'length']].to_csv('./barnes_and_noble_numeric_without_categoric.csv', index=False)

# df_2[['ID', 'UsedPrice', 'NewPrice', 'Pages','ISBN13', 'Publisher', 'height', 'width', 'length']].to_csv('./half_numeric.csv', index=False)
# df_2[['ID', 'UsedPrice', 'NewPrice', 'Pages', 'ISBN10', 'ISBN13', 'height', 'width', 'length']].to_csv('./half_numeric_without_categoric.csv', index=False)