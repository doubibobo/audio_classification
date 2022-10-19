from transformers import BertModel

import torch.nn as nn


class EmotionModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", first_token=True, is_frozen=False):
        super(EmotionModel, self).__init__()
        self.model_name = model_name
        self.bert_model = BertModel.from_pretrained(model_name, cache_dir=None)
        self.is_first_token = first_token
        if is_frozen:
            self.frozen_param()
            
        # self.lstm_layer = nn.GRU(
        #     768,
        #     768 // 2,
        #     2,
        #     batch_first=True,
        #     # dropout=drop_rate,
        #     bidirectional=True,
        # )
        # self.classify = nn.Sequential(nn.Linear(768, n_classes))
        
    def frozen_param(self):
        for _, parameter in self.bert_model.named_parameters():
            parameter.requires_grad = False

    def forward(self, ids):
        attention_mask = (ids > 0)
        # last seq 是最后一层的输出
        outputs = self.bert_model(input_ids=ids, attention_mask=attention_mask)
        # out = self.fc(last_seq[:, 0, :]).sigmoid()
        last_seq, pool_output = outputs[0], outputs[1]
        if self.is_first_token:
            return pool_output
        
        # TODO: 直接使用特征时，使用GRU进行规整
        # last_seq, _ = self.lstm_layer(last_seq)
        return last_seq
