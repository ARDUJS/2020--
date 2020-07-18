#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import config as user_config
import numpy as np
from sklearn.metrics import f1_score
from transformers import BertModel, BertConfig


# In[8]:


# class BertForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         self.init_weights()


# In[7]:


class classfiy(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, *arg):
        super(classfiy, self).__init__()
        self.result = {}
        num_labels, total_steps, _ = arg
        self.bert = BertModel.from_pretrained(user_config.pretrained)
        config = BertConfig.from_pretrained(user_config.pretrained)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.labels_data = None
        
#         #self.cls = BertOnlyMLMHead(config)
#         def init_weights(module):
#             if isinstance(module, (nn.Linear, nn.Embedding)):
#                 # Slightly different from the TF version which uses truncated_normal for initialization
#                 # cf https://github.com/pytorch/pytorch/pull/5617
#                 module.weight.data.normal_(mean=0.0, std=config.initializer_range)
#             elif isinstance(module, BERTLayerNorm):
#                 module.beta.data.normal_(mean=0.0, std=config.initializer_range)
#                 module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
#             if isinstance(module, nn.Linear):
#                 module.bias.data.zero_()
                
        def _init_weights(self, module):
            """ Initialize the weights """
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, BertLayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self, data, *arg):
        labels = None
        # print(len(data))
        if len(data) == 4:
            input_ids, attention_mask, token_type_ids, labels = data
        else:
            input_ids, attention_mask, token_type_ids = data
        if arg is not ():
            self.global_step, _ = arg
        self.labels_data = labels
        pooled_output = self.encoder(input_ids, attention_mask, token_type_ids)
        output = self.decoder(pooled_output)
        if labels is not None:
            loss = self.calc_loss(output, labels)
            return loss, torch.nn.functional.softmax(output, dim=-1)
        return torch.nn.functional.softmax(output, dim=-1)

    def encoder(self, *x):
        input_ids, attention_mask, token_type_ids = x
        all_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        return pooled_output

    def decoder(self, x):
        x = self.dropout(x)
        return self.classifier(x)

    def calc_loss(self, inputs, targets):
        loss_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        loss_cross_entropy = loss_cross_entropy(inputs, targets)
        self.result['loss'] = loss_cross_entropy.detach().item()
        self.result['accuracy'] = self.calc_accuracy(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        self.result['f1'] = self.calc_f1(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        return loss_cross_entropy

    def calc_accuracy(self, inputs, targets):
        outputs = np.argmax(inputs, axis=1)
        return np.mean(outputs == targets)

    def calc_f1(self, inputs, targets):
        outputs = np.argmax(inputs, axis=1)
        return f1_score(targets, outputs, average='macro')

    def get_result(self):
        return self.result

    def get_labels_data(self):
        return self.labels_data


# In[ ]:




