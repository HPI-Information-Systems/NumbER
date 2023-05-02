import sys
#sys.path.append('/Users/pradap/Documents/Research/Python-Package/anhaid/py_entitymatching/')

import py_entitymatching as em
import pandas as pd
import os

import sys
import torch
import numpy as np
from torch.utils import data
import random
import time
import pandas as pd
import deepmatcher as dm
from pathlib import Path

from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution

class MagellanMatchingSolution(MatchingSolution):
    def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
        super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
        
        
    def model_train(self, epochs, batch_size, pos_neg_ratio):
        A = em.read_csv_metadata(os.path.join(self.train_dataset_path, 'A.csv'), key='id')
        B = em.read_csv_metadata(os.path.join(self.train_dataset_path, 'B.csv'), key='id')
        ab = em.AttrEquivalenceBlocker()
        C1 = ab.block_tables(A, B, 'city', 'city', 
							l_output_attrs=['name', 'addr', 'city', 'phone'], 
							r_output_attrs=['name', 'addr', 'city', 'phone']
							)
        return best_f1, model, None, time.time() - start_time
        
    def model_predict(self, model):
        return {'predict': predictions, 'evaluate': f1}