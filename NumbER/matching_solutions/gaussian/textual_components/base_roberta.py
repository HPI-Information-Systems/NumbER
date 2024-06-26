from torch import nn
from transformers import RobertaModel, AutoTokenizer


class BaseRoberta(nn.Module):
    def __init__(self, max_length: int = 40, embedding_size: int = 256):
        super(BaseRoberta, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.roberta.to("cuda")
        self.embeddings = nn.Linear(self.roberta.config.hidden_size, embedding_size, device="cuda")
        self.dropout = nn.Dropout(0.1)
        self.max_length = max_length
        self.embedding_size = embedding_size
        #self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
    
    def get_outputshape(self):
        return self.embedding_size

    def forward(self, input_sequence):
        #if self.stage == Stage.PRETRAIN:
            #input_sequence = [torch.tensor(self.tokenizer.encode(input, max_length=self.max_length, truncation=True)) for input in input_sequence]
            #input_sequence = self.pad_tensors(input_sequence, self.max_length) #todo check if abstract
        outputs = self.roberta(
            input_sequence
        )
        #print("shape", np.shape(outputs[0][:, 0, :]))
        #return outputs[0][:, 0, :]
        embeddings = self.embeddings(outputs[0][:, 0, :])
        return embeddings
        #return pooled_output