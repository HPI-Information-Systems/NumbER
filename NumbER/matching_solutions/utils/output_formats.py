from abc import ABC
import pandas as pd
import pathlib
from tqdm import tqdm
import os
import itertools
import networkx as nx

class OutputFormat(ABC):
    def __init__(self, goldstandard, records, drop_id=True):
        self.goldstandard = goldstandard
        if drop_id:
            if "id" in records.columns:
                records.drop('id', axis=1, inplace=True)
        self.records = records
    def create_output(self):
        raise NotImplementedError
    #def write_to_file(self, filepath):
        #if os.path.exists(filepath):
        #    raise FileExistsError
class DittoFormat(OutputFormat):
    def __init__(self, goldstandard: pd.DataFrame, records: pd.DataFrame):
        super().__init__(goldstandard, records, True)
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
        return result, None
    def write_to_file(self, filepath):
        filepath = f"{filepath}.txt" if filepath[-4:] != ".txt" else filepath
        with open(filepath, "w") as file:
            output = self.create_output()[0]
            for element in output:
                file.write(element + "\n")
        return filepath, None
    
class EmbittoFormat(OutputFormat):
    def __init__(self, goldstandard: pd.DataFrame, records: pd.DataFrame):
        super().__init__(goldstandard, records, False)
        self.goldstandard = goldstandard
        self.records = records
        
    def create_output(self):
        all_ids = set(self.goldstandard['p1'].values).union(set(self.goldstandard['p2'].values))
        records = self.records[self.records['id'].isin(all_ids)]
        return records, self.goldstandard
    
    def write_to_file(self, filepath):
        filepath = f"{filepath}.csv" if filepath[-4:] != ".csv" else filepath
        records, goldstandard = self.create_output()
        records.to_csv(filepath, index=False)
        goldstandard.to_csv(f"{filepath[:-4]}_goldstandard.csv", index=False)
        print("WRITING TO FILE", filepath)
        return filepath, f"{filepath[:-4]}_goldstandard.csv"

class DeepMatcherFormat(OutputFormat):
    def __init__(self, goldstandard: pd.DataFrame, records: pd.DataFrame):
        super().__init__(goldstandard, records, True)
        self.goldstandard = goldstandard
        self.records = records
        
    def create_output(self):
        left_columns = [f"left_{column}" for column in self.records.columns]
        right_columns = [f"right_{column}" for column in self.records.columns]
        result = []
        for _, pair in tqdm(self.goldstandard.iterrows()):
            result.append([pair['prediction']] + list(self.records.loc[pair['p1']]) + list(self.records.loc[pair['p2']]))
        df = pd.DataFrame(result, columns=['prediction'] + left_columns + right_columns)
        df['id'] = df.index
        return df, None
    
    def write_to_file(self, filepath):
        print("WRITING TO FILE", filepath)
        filepath = f"{filepath}.csv" if filepath[-4:] != ".csv" else filepath
        self.create_output()[0].to_csv(filepath, index=False)
        return filepath, None

class MD2MFormat(OutputFormat):
    def __init__(self, goldstandard: pd.DataFrame, records: pd.DataFrame):
        self.goldstandard = goldstandard
        self.records = records
        
    def create_output(self):
        #goldstandard: pair1, pair2, prediction
        #self.records.rename({'prediction': 'label'}, axis=1, inplace=True)
        goldstandard = pd.DataFrame(self.transitively_close(self.goldstandard), columns=['p1', 'p2'])
        goldstandard['prediction'] = 1
        list_of_ids = list(set(goldstandard['p1'].unique()).union(set(goldstandard['p2'].unique())))
        print("LIST OF IDS.", list_of_ids)
        records = self.records[self.records['id'].isin(list_of_ids)]
        return goldstandard, records
        #matches = set(list(tuple(sorted((goldstandard['p1'], goldstandard['p2'])))))
        
    def transitively_close(self,pairs):
        pairs = pairs[pairs['prediction'] == 1]
        G = nx.from_pandas_edgelist(pairs, 'p1', 'p2', create_using=nx.DiGraph())
        G_tc = nx.transitive_closure(G)
        df_tc = pd.DataFrame(list(G_tc.edges()), columns=['p1', 'p2'])
        df_tc['tuple'] = df_tc.apply(lambda row: tuple(sorted((row['p1'], row['p2']))), axis=1)
        list_of_ids = list(set(df_tc['p1'].unique()).union(set(df_tc['p2'].unique())))
        reflexive = [tuple(sorted((val, val))) for val in list_of_ids]
        print("Reflexive", reflexive)
        df_tc = set(df_tc['tuple']).union(set(reflexive))
        return list(df_tc)
    
    def write_to_file(self, filepath):
        #try:
        #super().write_to_file(filepath)
        #except FileExistsError:
        #    return filepath
        filepath = f"{filepath}.csv" if filepath[-4:] != ".csv" else filepath
        goldstandard, records = self.create_output()
        records.to_csv(f"{filepath[:-4]}_records.csv", index=False)
        #if records is not None:
        goldstandard.to_csv(filepath, index=False)
        return filepath, f"{filepath[:-4]}_records.csv"

class DummyFormat(OutputFormat):
    def __init__(self, goldstandard, records, output_format: str, base_data_path: str):
        super().__init__(goldstandard, records, False)#todo check if true or false
        self.output_format = output_format
        self.base_data_path = base_data_path
        
    def write_to_file(self, filepath):
        os.makedirs(self.base_data_path, exist_ok=True)
        filename = pathlib.Path(filepath).name
        print("Filename", filename)
        fileending = "txt" if self.output_format == "ditto" else "csv"
        filepath = os.path.join(self.base_data_path, f"{self.output_format}_{filename}.{fileending}")
        goldstandard_path = os.path.join(self.base_data_path, f"{filename.replace(self.output_format+'_', '')}_goldstandard.csv")  
        if not os.path.exists(goldstandard_path):
            if "index" in self.goldstandard.columns:
                self.goldstandard.drop('index', axis=1, inplace=True)
            self.goldstandard.to_csv(goldstandard_path, index=False)
        else:
            self.goldstandard = pd.read_csv(goldstandard_path)
        if not os.path.exists(filepath):
            mapping = {
                'ditto': DittoFormat,
                'deep_matcher': DeepMatcherFormat,
                'embitto': EmbittoFormat,
                'lightgbm': DeepMatcherFormat,
            }
            return mapping[self.output_format](self.goldstandard, self.records).write_to_file(os.path.join(pathlib.Path(filepath).parent, pathlib.Path(filepath).stem))
        
        return filepath, goldstandard_path if self.output_format == "embitto" else None