from NumbER.matching_solutions.embitto.embitto import Embitto
from NumbER.matching_solutions.embitto.data_loader import EmbittoDataModule
import pytorch_lightning as pl
import pandas as pd

def train():
    train_data = pd.read_csv(self.train_path)
    valid_data = pd.read_csv(self.valid_path)
    test_data = pd.read_csv(self.test_path)
    numerical_train_data, textual_train_data, train_matches = process_dataframe(train_data)
    numerical_valid_data, textual_valid_data, valid_matches = process_dataframe(valid_data)
    numerical_test_data, textual_test_data, test_matches = process_dataframe(test_data)
    embitto = Embitto()
    data = EmbittoDataModule()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(embitto, data)
    
def get_numeric_values(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    return df.select_dtypes(include=numerics)

def get_textual_values(df):
    data = df[get_textual_columns(df)]
    result = data.copy()
    for idx, record in data.iterrows():
        temp = ""
        for col, val in record.iteritems():
            temp += f"COL {col} VAL {val}"
        result.loc[idx] = temp
    return result
    #input: die record liste.

def get_textual_columns(df):
    data = df.select_dtypes(include=['object'])
    return data.columns

def process_dataframe(df):
    matches = df['prediction']
    df.drop(columns=['prediction'], inplace=True)
    numerical_data = get_numeric_values(df)
    textual_data = get_textual_values(df)
    return {'numerical_data': numerical_data, 'textual_data': textual_data, 'matches': matches}
