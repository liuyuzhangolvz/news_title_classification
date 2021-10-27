import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class BertClassify(nn.Module):
    def __init__(self, bert, label_size, hidden_size, hidden_dropout_prob):
        super(BertClassify, self).__init__()
        self.bert = bert
        self.linear = nn.Linear(hidden_size, label_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #print('outputs type >>', type(outputs))
        #print('outputs last_hidden_state >>', outputs.last_hidden_state.size())
        #print('outputs pooler_output >>', outputs.pooler_output.size())

        pooled_out = self.dropout(outputs.pooler_output)
        #print('pooled_out>>', pooled_out.size())

        logits = self.linear(pooled_out)
        #print('logits size>>', logits.size())
        #print('logits>>', logits)
        if labels is None:
            return logits
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        #print('loss>>', loss)

        return (logits, ) + (loss, )