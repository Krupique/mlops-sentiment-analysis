# Pretrained DistilBERT model wrapper
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, transformer, num_labels = 6):
        super(Model, self).__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(transformer.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids):
        # Getting the output of sequence of transformer
        outputs = self.transformer(input_ids)
        sequence_output = outputs.last_hidden_state
        
        # Selecting the fist token of each sequence (token CLS from BERT) to classification
        cls_token = sequence_output[:, 0, :]

        # Adding a dense layer to the output with softmax activation for classification
        logits = self.classifier(cls_token)
        out = self.softmax(logits)

        return out