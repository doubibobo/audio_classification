import torch
from transformers import RobertaModel

import torch.nn as nn


class RobertaModelEmotion(nn.Module):
    def __init__(self, model_name="roberta-base", first_token=True, is_frozen=False):
        super(RobertaModelEmotion, self).__init__()
        self.model_name = model_name
        self.bert_model = RobertaModel.from_pretrained(model_name, cache_dir=None)
        self.is_first_token = first_token
        if is_frozen:
            self.frozen_param()
        # self.classify = nn.Sequential(nn.Linear(768, n_classes))
    
    def frozen_param(self):
        for _, parameter in self.bert_model.named_parameters():
            parameter.requires_grad = False

    def forward(self, ids):
        # TODO 要使用roberta，这里的ID要换成RobertaTokenizer生成的ID，我咋感觉这里应该是ids != 0
        attention_mask = ids != 1
        # last seq 是最后一层的输出
        outputs = self.bert_model(input_ids=ids, attention_mask=attention_mask)
        # out = self.fc(last_seq[:, 0, :]).sigmoid()
        last_seq, pool_output = outputs[0], outputs[1]
        if self.is_first_token:
            return pool_output
        return last_seq


if __name__ == "__main__":
    # model = RobertaModelEmotion()
    # test_data = torch.tensor([i for i in range(32 * 256)]).reshape((32, 256))
    # output = model(test_data)
    pass
