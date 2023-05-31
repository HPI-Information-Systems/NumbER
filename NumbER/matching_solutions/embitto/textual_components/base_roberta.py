from torch import nn
from transformers import RobertaModel, AutoTokenizer

class RobertaClassifier(nn.Module):
    def __init__(self):
        super(RobertaClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        #self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_sequence, attention_mask=None, token_type_ids=None, labels=None):
        #input: textual_columns, label
        # finetuning: s1, s2, label
        #das muss in die dataset preparation eigentlich
        input_ids = self.tokenizer.encode(input_sequence, add_special_tokens=True)
        outputs = self.roberta(
            input_ids
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits