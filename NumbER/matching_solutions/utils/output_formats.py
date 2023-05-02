from abc import ABC
import pandas as pd
from tqdm import tqdm
import os

class OutputFormat(ABC):
    def __init__(self, goldstandard, records):
        self.records = records
        self.goldstandard = goldstandard
    def create_output(self):
        raise NotImplementedError
    def write_to_file(self, filepath):
        if os.path.exists(filepath):
            raise FileExistsError
class DittoFormat(OutputFormat):
    def __init__(self, goldstandard: pd.DataFrame, records: pd.DataFrame):
        self.goldstandard = goldstandard
        self.records = records
        
    def create_output(self):
           #goldstandard: pair1, pair2, prediction
        result = []
        for _, pair in tqdm(self.goldstandard.iterrows()):
            temp=""
            record1 = self.records.loc[pair['p1']]
            record2 = self.records.loc[pair['p2']]
            for col, val in record1.iteritems():
                temp += f" COL {col} VAL {val}"
            temp += "\t"
            for col, val in record2.iteritems():
                temp += f" COL {col} VAL {val}"
            temp += "\t"
            temp += str(pair['prediction'])
            result.append(temp)
        return result
    def write_to_file(self, filepath):
        filepath = f"{filepath}.txt" if filepath[-4:] != ".txt" else filepath
        try:
            super().write_to_file(filepath)
        except FileExistsError:
            return filepath
        with open(filepath, "w") as file:
            for element in self.create_output():
                file.write(element + "\n")
        return filepath

class DeepMatcherFormat(OutputFormat):
    def __init__(self, goldstandard: pd.DataFrame, records: pd.DataFrame):
        self.goldstandard = goldstandard
        self.records = records
        
    def create_output(self):
           #goldstandard: pair1, pair2, prediction
        #self.records.rename({'prediction': 'label'}, axis=1, inplace=True)
        left_columns = [f"left_{column}" for column in self.records.columns]
        right_columns = [f"right_{column}" for column in self.records.columns]
        result = []
        for _, pair in tqdm(self.goldstandard.iterrows()):
            result.append([pair['prediction']] + list(self.records.loc[pair['p1']]) + list(self.records.loc[pair['p2']]))
        df = pd.DataFrame(result, columns=['prediction'] + left_columns + right_columns)
        df['id'] = df.index
        return df
    
    def write_to_file(self, filepath):
        try:
            super().write_to_file(filepath)
        except FileExistsError:
            return filepath
        filepath = f"{filepath}.csv" if filepath[-4:] != ".csv" else filepath
        self.create_output().to_csv(filepath, index=False)
        return filepath