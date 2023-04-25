import sys
import torch
import numpy as np
from torch.utils import data
import random
import time

sys.path.insert(0, "Snippext_public")

from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution
from NumbER.matching_solutions.ditto.ditto_light.dataset import DittoDataset
from NumbER.matching_solutions.ditto.ditto_light.summarize import Summarizer
from NumbER.matching_solutions.ditto.ditto_light.knowledge import *
from NumbER.matching_solutions.ditto.ditto_light.ditto import train, evaluate
from NumbER.matching_solutions.ditto.matcher import predict

class DittoMatchingSolution(MatchingSolution):
    def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
        super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
        self.file_format = 'ditto'
        
    def model_train(self, run_id, batch_size, max_len, lr, n_epochs, lm, fp16, size=None, da=None, dk=None, summarize=None):
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_id)
        
        run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (self.dataset_name, lm, da,
                dk, summarize, str(size), run_id)
        run_tag = run_tag.replace('/', '_')
        
        #todo make summarize working
        if summarize:
            raise NotImplementedError
        # if summarize:
        #     summarizer = Summarizer(config, lm=lm)
        #     trainset = summarizer.transform_file(trainset, max_len=max_len)
        #     validset = summarizer.transform_file(validset, max_len=max_len)
        #     testset = summarizer.transform_file(testset, max_len=max_len)

        # if dk is not None:
        #     if dk == 'product':
        #         injector = ProductDKInjector(config, dk)
        #     else:
        #         injector = GeneralDKInjector(config, dk)

        #     trainset = injector.transform_file(trainset)
        #     validset = injector.transform_file(validset)
        #     testset = injector.transform_file(testset)

        # load train/dev/test sets
        train_dataset = DittoDataset(self.train_path,
                                    lm=lm,
                                    max_len=max_len,
                                    size=size,
                                    da=da)
        valid_dataset = DittoDataset(self.valid_path, lm=lm)
        #test_dataset = DittoDataset(testset, lm=lm)
        start_time = time.time()
        best_f1, model, threshold = train(train_dataset, valid_dataset, batch_size, lm, lr, n_epochs, fp16)
        return best_f1, model, threshold, time.time() - start_time
        
    def model_predict(self, model, batch_size, lm, max_len,summarizer=None, dk_injector=None, threshold=None):
        test_dataset = DittoDataset(self.test_path, lm=lm)
        test_iter = data.DataLoader(dataset=test_dataset,
                                 batch_size=batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=test_dataset.pad) #stimmt das so?
        f1, _ = evaluate(model, test_iter)
        return {'predict': predict(self.test_path, self.dataset_name, model, batch_size, summarizer, lm, max_len, dk_injector, threshold), 'evaluate': f1}
    
