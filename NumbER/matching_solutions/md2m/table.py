import typing
import pandas as pd

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
        #print(a)
        #print("S", b.at['longitude'])
        #a,b sind zwei records
        sum = 0
        maximum_p_c = max(self.comparison_cols, key=lambda x: x.p_c).p_c
        for column in self.comparison_cols:
            a_value = a.at[column.name]
            b_value = b.at[column.name]
            print(column.p_c)
            sum += (column.p_c/maximum_p_c) * column.column_similarity(a_value,b_value)
        return sum / len(self.comparison_cols)

    def calculate_candidates(self):
        result = []
        for idx_a, row_a in self.data.iterrows():
            for idx_b, row_b in self.data.iterrows():
                if idx_a != idx_b:
                    sim = self.similarity(row_a, row_b)
                    print("h",sim)
                    result.append({'p1': idx_a, 'p2': idx_b, 'sim': sim})
        return pd.DataFrame(result, index=None)
                    
                
                   