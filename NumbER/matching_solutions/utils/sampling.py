from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

def sorted_neighbourhood(records, attributes, id_column, size):
	records = records.sort_values(by=attributes)[id_column].to_numpy()
	print(records)
	sliding_windows = sliding_window_view(records, size)
	pairs = set()
	for window in sliding_windows:
		for i in range(len(window)):
			for j in range(i+1, len(window)):
				pairs.add(tuple(sorted((window[i], window[j]))))
	return pairs

def map_to_groundtruth(pairs, gt):
    if 'prediction' in gt.columns:
        gt = gt[gt['prediction'] == 1]
    gt = gt[['p1', 'p2']].to_numpy()
    gt = set(tuple(sorted(pair)) for pair in gt)
    #print(pairs)
    res = []
    for pair in pairs:
        if pair in gt:
            print("Y")
            res.append(1)
        else:
            res.append(0)
    return res

    #1 if pair in gt else 0 for pair in pairs]


#sample records

#df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'b': [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'c': [5, 2, 1, 3,2, 3, 4, 5, 6, 7, 8]})
#print(sorted_neighbourhood(df, ['b', 'c'], 'a', 3))