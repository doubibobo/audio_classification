import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """
    参考自: https://www.kaggle.com/code/quincyqiang/feedback-meanpoolingv2-inference
    """
    def __init__(self, config):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(last_hidden_state.shape[0: -1])
            attention_mask = attention_mask.to(last_hidden_state.device)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
