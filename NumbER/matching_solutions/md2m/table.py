import typing
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
from NumbER.matching_solutions.md2m.column import Column

class Table():
    def __init__(self, name: str, cols, data: pd.DataFrame) -> None:
        self.name = name
        comparison_column_threshold = 0.05
        def is_above_threshold(el: Column):
            #print(el.p_c)
            return True if el.p_c > comparison_column_threshold else False
        self.comparison_cols = list(filter(is_above_threshold, cols))
        self.data = data
    
    def similarity(self, a, b):
        sum = 0
        maximum_p_c = max(self.comparison_cols, key=lambda x: x.p_c).p_c
        for column in self.comparison_cols:
            a_value = a.at[column.name]
            b_value = b.at[column.name]
            if np.isnan(a_value) or np.isnan(b_value): # type(a_value) != float and type(a_value) != int or type(b_value) != float and type(b_value) != int:
                continue
            sum += (column.p_c/maximum_p_c) * column.column_similarity(a_value,b_value)
        return sum / len(self.comparison_cols)

    def calculate_candidates(self):
        result = []
        print(len(self.data))
        combinations = itertools.combinations(self.data.index, 2)
        for i,j in tqdm(combinations):
            row_a = self.data.loc[i]
            row_b = self.data.loc[j]
            #if i != j:
            sim = self.similarity(row_a, row_b)
            result.append({'p1': i, 'p2': j, 'sim': sim})
        return pd.DataFrame(result, index=None)
                    
                
                   