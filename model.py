import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class BertClassify(nn.Module):
    def __init__(self, bert, label_size, hidden_size, hidden_dropout_prob):
        super(BertClassify, self).__init__()
        self.bert = bert
        self.linear = nn.Linear(hidden_size, label_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_out = self.dropout(outputs.pooler_output)

        logits = self.linear(pooled_out)

        if labels is None:
            return logits
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return (logits, ) + (loss, )


class TextCNN(nn.Module):
    def __init__(self, label_size, hidden_size, encode_layer=12, num_filters=3, filter_sizes=[2, 2, 2]):
        super(TextCNN, self).__init__()
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.encode_layer =encode_layer
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_filter_total = num_filters * len(filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, label_size)
        self.bias = nn.Parameter(torch.ones([label_size]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters,
                                                    kernel_size=(size, hidden_size)) for size in filter_sizes])
    def forward(self, x):
        """
        input:
            x: (batch_size, encode_layer, hidden_dim)
        output:
            output: (batch_size, label_size)
        """
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim)
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))
            mp = nn.MaxPool2d(kernel_size=(self.encode_layer - self.filter_sizes[i] + 1, 1))

            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)
        h_pool = torch.cat(pooled_outputs, len(self.filter_list))
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
        output = self.Weight(h_pool_flat) + self.bias  # (batch_size, label_size)
        return output


class Bert_Blend_CNN(nn.Module):
    def __init__(self, bert, label_size, hidden_size, num_hidden_layers=12, num_filters=3, filter_sizes=[2, 2, 2]):
        super(Bert_Blend_CNN, self).__init__()
        self.bert = bert
        self.linear = nn.Linear(hidden_size, label_size)
        self.textcnn = TextCNN(label_size, hidden_size, encode_layer=num_hidden_layers, num_filters=num_filters, filter_sizes=[2, 2, 2])

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        hidden_states = outputs.hidden_states  # bert_layers * (batch_size, seq_len, hidden_dim)
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # (batch_size, 1, hidden_dim)
        for i in range(2, len(hidden_states)):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # print('cls_embeddings>>', cls_embeddings.shape)

        logits = self.textcnn(cls_embeddings)

        if labels is None:
            return logits
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        #print('loss>>', loss)

        return (logits, ) + (loss, )