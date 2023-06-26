import numpy as np
import pandas as pd
import sys
import math
from NumbER.matching_solutions.embitto.enums import Phase, Stage
from NumbER.matching_solutions.embitto.formatters import dummy_formatter, pair_based_numeric_formatter
import torch

class DICEEmbeddings():
    def __init__(self, config, d):
        self.embeddings = []
        for col in config:
            self.embeddings.append(DICE(d=d, min_bound=col["lower_bound"], max_bound=col["upper_bound"]))
            
    def __call__(self, numbers):
        return self.calculate_embeddings(numbers)
    
    def calculate_embeddings(self, batch):
        res = []
        for numbers in batch:
            embed = []
            for idx_2, number in enumerate(numbers):
                if isinstance(number, tuple):
                    embedding_1 = self.embeddings[idx_2].make_dice(number[0])
                    embedding_2 = self.embeddings[idx_2].make_dice(number[1])
                    embeddings = np.subtract(embedding_1, embedding_2) 
                    #embeddings = np.concatenate((embedding_1,embedding_2))
                else:
                    embeddings = self.embeddings[idx_2].make_dice(number)
                embed.append(embeddings)
            res.append(np.concatenate(embed))
        return torch.tensor(res).to("cuda")
    
class DICE:
    '''
    copied from: https://github.com/wjdghks950/Methods-for-Numeracy-Preserving-Word-Embeddings/tree/master
    DICE class turns numbers into their respective DICE embeddings
    
    Since the cosine function decreases monotonically between 0 and pi, simply employ a linear mapping
    to map distances s_n \in [0, |a-b|] to angles \theta \in [0, pi]
    '''
    def __init__(self, d=2, min_bound=0, max_bound=100, norm="l2"):
        self.d = d # By default, we build DICE-2
        self.min_bound = min_bound
        self.max_bound = max_bound
        if self.min_bound == self.max_bound: #todo does this make sense
            self.min_bound = self.min_bound - 1
            self.max_bound = self.max_bound + 1
        self.norm = norm  # Restrict x and y to be of unit length
        self.M = np.random.normal(0, 1, (self.d, self.d))
        self.Q, self.R = np.linalg.qr(self.M, mode="complete")  # QR decomposition for orthonormal basis, Q
    
    def __linear_mapping(self, num):
        '''Eq. (4) from DICE'''
        norm_diff = num / abs(self.min_bound - self.max_bound)
        theta = norm_diff * math.pi
        return theta
    
    def make_dice(self, num):
        if np.isnan(num):
            return [0]*self.d
        r = 1
        theta = self.__linear_mapping(num)
        if self.d == 2:
            # DICE-2
            polar_coord = np.array([r*math.cos(theta), r*math.sin(theta)])
        elif self.d > 2:
            # DICE-D
            polar_coord = np.array([math.sin(theta)**(dim-1) * math.cos(theta) if dim < self.d else math.sin(theta)**(self.d) for dim in range(1, self.d+1)])
        else:
            raise ValueError("Wrong value for `d`. `d` should be greater than or equal to 2.")
            
        dice = np.dot(self.Q, polar_coord)  # DICE-D embedding for `num`
        
        # return dice.tolist()
        return dice
    
class DICEEmbeddingAggregator():
    def __init__(self, train_dice:DICEEmbeddings, valid_dice: DICEEmbeddings, test_dice: DICEEmbeddings):
        self.train_dice = train_dice
        self.valid_dice = valid_dice
        self.test_dice = test_dice
    
    def __call__(self, numbers, phase: Phase):
        if phase == phase.TEST:
            return self.test_dice(numbers)
        elif phase == phase.TRAIN:
            return self.train_dice(numbers)
        elif phase == phase.VALID:
            return self.valid_dice(numbers)
    @staticmethod
    def get_formatter(stage: Stage):
        if stage == stage.PRETRAIN:
            return dummy_formatter
        elif stage == stage.FINETUNING:
            return pair_based_numeric_formatter