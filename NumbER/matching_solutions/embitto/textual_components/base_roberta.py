from torch import nn
from transformers import RobertaModel, AutoTokenizer
from NumbER.matching_solutions.embitto.embitto import Stage
import torch
class RobertaClassifier(nn.Module):
    def __init__(self, max_length, stage: Stage):
        super(RobertaClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.embeddings = nn.Linear(self.roberta.config.hidden_size, 256)
        self.dropout = nn.Dropout(0.1)
        self.max_length = max_length
        self.stage = stage
        #self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_sequence):
        #input: textual_columns, label
        # finetuning: s1, s2, label
        #das muss in die dataset preparation eigentlich
        input_ids = []
        if self.stage == Stage.PRETRAIN:
            for input in input_sequence:
            #input_ids = torch.tensor(self.tokenizer.encode(input_sequence, return_tensors="pt",max_length=40, truncation=True)).to('cuda')
                input_ids.append(torch.tensor(self.tokenizer.encode(input, max_length=40, truncation=True)))#.to('cuda'))
            input_ids = torch.squeeze(torch.stack(input_ids)).to('cuda')
                #textual_data = self.tokenizer.encode(self.textual_data[idx],return_tensors="pt")
            #input_ids = torch.stack(input_ids)
        # else:
        #     input_ids = self.tokenizer.encode(input_sequence[0], input_sequence[1], add_special_tokens=True)
        outputs = self.roberta(
            input_ids
        )
        #return outputs[0][:, 0, :]
        embeddings = self.embeddings(outputs[0][:, 0, :])
        #print("M", embeddings)
        return embeddings
        
        #return pooled_output