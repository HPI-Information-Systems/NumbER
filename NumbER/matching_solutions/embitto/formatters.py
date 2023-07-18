import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler
number_pattern = re.compile(
    r"(\d+)\.?(\d*)")  # Matches numbers in decimal form.
def ditto_formatter(data, scientific_notation=False):
    if "id" in data.columns:
        data.drop(columns=['id'], inplace=True)
    result = []
    for _, record in data.iterrows():
        temp = ""
        for col, val in record.items():
            temp += f"COL {col} VAL {val} "
        temp = apply_scientific_notation(temp) if scientific_notation else temp
        result.append(temp)
    return result

def pair_based_ditto_formatter(data, scientific_notation=False):
    columns = data.columns
    left_columns = filter(lambda x: x.startswith("left_"), columns)
    right_columns = filter(lambda x: x.startswith("right_"), columns)
    left_data = data[left_columns]
    right_data = data[right_columns]
    left_data.columns = left_data.columns.str.replace('left_', '')
    right_data.columns = right_data.columns.str.replace('right_', '')
    left_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    right_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    left_data = ditto_formatter(left_data, scientific_notation)
    right_data = ditto_formatter(right_data, scientific_notation)
    return list(zip(left_data, right_data))

def pair_based_numeric_formatter(data):
    columns = data.columns
    left_columns = filter(lambda x: x.startswith("left_"), columns)
    right_columns = filter(lambda x: x.startswith("right_"), columns)
    left_data = data[left_columns]
	#rename columns in dataframe by removing the prefix left_ and right_ from column names
    right_data = data[right_columns]
    left_data.columns = left_data.columns.str.replace('left_', '')
    right_data.columns = right_data.columns.str.replace('right_', '')
    left_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    right_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    df = {}
    for col in left_data.columns:
        df[col] = list(zip(left_data[col].values, right_data[col].values))
    df = pd.DataFrame(df)
    return df

def numeric_prompt_formatter(data):
    columns = data.columns
    left_columns = filter(lambda x: x.startswith("left_"), columns)
    right_columns = filter(lambda x: x.startswith("right_"), columns)
    left_data = data[left_columns]
	#rename columns in dataframe by removing the prefix left_ and right_ from column names
    right_data = data[right_columns]
    left_data.columns = left_data.columns.str.replace('left_', '')
    right_data.columns = right_data.columns.str.replace('right_', '')
    left_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    right_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    df = {}
    for col in left_data.columns:
        df[col] = list(zip(left_data[col].values, right_data[col].values))
    df = pd.DataFrame(df)
    data = []
    for _, row in df.iterrows():
        temp = ""
        for col, val in row.items():
            temp += f"COL {col} L_VAL {val[0]} R_VAL {val[1]} DIST {abs(val[0] - val[1])} "
        data.append(temp)
    return data

def complete_prompt_formatter(data, scientific_notation=False, min_max_scaled=False, train_data=None):
    columns = data.columns
    print("data2", data)
    left_columns = filter(lambda x: x.startswith("left_"), columns)
    right_columns = filter(lambda x: x.startswith("right_"), columns)
    left_data = data[left_columns]
	#rename columns in dataframe by removing the prefix left_ and right_ from column names
    right_data = data[right_columns]
    left_data.columns = left_data.columns.str.replace('left_', '')
    right_data.columns = right_data.columns.str.replace('right_', '')
    left_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    right_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = list(left_data.select_dtypes(include=numerics).columns)
    if min_max_scaled:
        for col in numeric_cols:
            scaler = MinMaxScaler()
            print(train_data)
            left_scaler = scaler.fit(train_data[col].values.reshape(-1,1))
            left_data[col] = left_scaler.transform(left_data[col].values.reshape(-1,1)).reshape(-1)
            right_scaler = scaler.fit(train_data[col].values.reshape(-1,1))
            right_data[col] = right_scaler.transform(right_data[col].values.reshape(-1,1)).reshape(-1)
    df = {}
    for col in left_data.columns:
        df[col] = list(zip(left_data[col].values, right_data[col].values))
    df = pd.DataFrame(df)
    data_left = []
    data_right = []
    for _, row in df.iterrows():
        temp_left = ""
        temp_right = ""
        for col, val in row.items():
            if col in numeric_cols:
                dist = abs(val[0] - val[1])
                temp_left += f"COL {col} VAL {float(val[0])} DIST {float(dist)} "
                temp_right += f"COL {col} VAL {float(val[1])} DIST {float(dist)} "
            else:
                temp_left += f"COL {col} VAL {val[0]} "
                temp_right += f"COL {col} VAL {val[1]} "
        temp_left = apply_scientific_notation(temp_left) if scientific_notation else temp_left
        temp_right = apply_scientific_notation(temp_right) if scientific_notation else temp_right
        data_left.append(temp_left)
        data_right.append(temp_right)
    return list(zip(data_left, data_right))

def textual_prompt_formatter(data):
    columns = data.columns
    left_columns = filter(lambda x: x.startswith("left_"), columns)
    right_columns = filter(lambda x: x.startswith("right_"), columns)
    left_data = data[left_columns]
	#rename columns in dataframe by removing the prefix left_ and right_ from column names
    right_data = data[right_columns]
    left_data.columns = left_data.columns.str.replace('left_', '')
    right_data.columns = right_data.columns.str.replace('right_', '')
    left_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    right_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = list(left_data.select_dtypes(include=numerics).columns)
    df = {}
    for col in left_data.columns:
        df[col] = list(zip(left_data[col].values, right_data[col].values))
    df = pd.DataFrame(df)
    data = []
    for _, row in df.iterrows():
        temp = ""
        for col, val in row.items():
            if col in numeric_cols:
                temp += f"COL {col} L_VAL {float(val[0])} R_VAL {float(val[1])} DIST {abs(float(val[0]) - float(val[1]))} "
            else:
                temp += f"COL {col} L_VAL {val[0]} R_VAL {val[1]} "
        data.append(temp)
    return data
    
def dummy_formatter(data):
	return data

def complete_prompt_formatter_scientific(data):
    return complete_prompt_formatter(data, scientific_notation=True)

def complete_prompt_formatter_min_max_scaled(data, train_data):
    print(train_data)
    return complete_prompt_formatter(data, min_max_scaled=True, train_data=train_data)

def pair_based_ditto_formatter_scientific(data):
    return pair_based_ditto_formatter(data, scientific_notation=True)

def text_sim_formatter(data, train_data):
    columns = data.columns
    left_columns = filter(lambda x: x.startswith("left_"), columns)
    right_columns = filter(lambda x: x.startswith("right_"), columns)
    left_data = data[left_columns]
	#rename columns in dataframe by removing the prefix left_ and right_ from column names
    right_data = data[right_columns]
    left_data.columns = left_data.columns.str.replace('left_', '')
    right_data.columns = right_data.columns.str.replace('right_', '')
    left_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    right_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = list(left_data.select_dtypes(include=numerics).columns)
    dists = {}
    for col in numeric_cols:
        dist = pdist(train_data[col].values, 'euclidean')
        very_high_dist = np.quantile(dist, 0.95)
        high_dist = np.quantile(dist, 0.75)
        very_low_dist = np.quantile(dist, 0.05)
        low_dist = np.quantile(dist, 0.25)
        dists[col] = {
            'very_high_dist': very_high_dist,
            'high_dist': high_dist,
            'very_low_dist': very_low_dist,
            'low_dist': low_dist
        }
    df = {}
    for col in left_data.columns:
        df[col] = list(zip(left_data[col].values, right_data[col].values))
    df = pd.DataFrame(df)
    data_left = []
    data_right = []
    for _, row in df.iterrows():
        temp_left = ""
        temp_right = ""
        for col, val in row.items():
            if col in numeric_cols:
                dist = abs(val[0] - val[1])
                if dist <= dists[col]['very_low_dist']:
                    dist = "very low"
                elif dist <= dists[col]['low_dist']:
                    dist = "low"
                elif dist <= dists[col]['high_dist']:
                    dist = "high"
                elif dist <= dists[col]['very_high_dist']:
                    dist = "very high"
                temp_left += f"COL {col} VAL {float(val[0])} DIST {dist} "
                temp_right += f"COL {col} VAL {float(val[1])} DIST {dist} "
            else:
                temp_left += f"COL {col} VAL {val[0]} "
                temp_right += f"COL {col} VAL {val[1]} "
        data_left.append(temp_left)
        data_right.append(temp_right)
    return list(zip(data_left, data_right))
    

def number_repl(matchobj):
  """Given a matchobj from number_pattern, it returns a string writing the corresponding number in scientific notation."""
  pre = matchobj.group(1).lstrip("0")
  post = matchobj.group(2)
  if pre and int(pre):
    # number is >= 1
    exponent = len(pre) - 1
  else:
    # find number of leading zeros to offset.
    exponent = -re.search("(?!0)", post).start() - 1
    post = post.lstrip("0")
  return (pre + post).rstrip("0") + " scinotexp " + str(exponent)

def apply_scientific_notation(line):
  """Convert all numbers in a line to scientific notation."""
  #TODO(ramachandrand): If the number is already in the vocab should we keep as is?
  return re.sub(number_pattern, number_repl, line)