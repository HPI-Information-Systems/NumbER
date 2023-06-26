import pandas as pd

def ditto_formatter(data):
    result = []
    for _, record in data.iterrows():
        temp = ""
        for col, val in record.items():
            temp += f"COL {col} VAL {val} "
        result.append(temp)
    return result

def pair_based_ditto_formatter(data):
    columns = data.columns
    left_columns = filter(lambda x: x.startswith("left_"), columns)
    right_columns = filter(lambda x: x.startswith("right_"), columns)
    left_data = data[left_columns]
    right_data = data[right_columns]
    left_data.columns = left_data.columns.str.replace('left_', '')
    right_data.columns = right_data.columns.str.replace('right_', '')
    left_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    right_data.drop(columns=['id'], inplace=True) if "id" in left_data.columns else None
    left_data = ditto_formatter(left_data)
    right_data = ditto_formatter(right_data)
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
    return pd.DataFrame(df)
    
def dummy_formatter(data):
	return data