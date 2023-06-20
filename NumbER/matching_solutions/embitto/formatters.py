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
	#rename columns in dataframe by removing the prefix left_ and right_ from column names
	left_data.columns = left_data.columns.str.replace('left_', '')
	right_data.columns = right_data.columns.str.replace('right_', '')
	left_data = ditto_formatter(left_data)
	right_data = ditto_formatter(right_data)
	return list(zip(left_data, right_data))