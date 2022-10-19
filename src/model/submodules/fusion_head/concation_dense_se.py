import torch.nn as nn
from src.model.submodules.fusion_head import SENet
import torch.nn.functional as F


class ConcatDenseSE(nn.Module):

    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        # self.ln = nn.LayerNorm(multimodal_hidden_size)
        # self.bn = nn.BatchNorm1d(multimodal_hidden_size)
        # self.fc1 = nn.Linear(multimodal_hidden_size, hidden_size)
        # self.drop1 = nn.Dropout(dropout)
        hidden_size = multimodal_hidden_size
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, embedding):
        ""
        # embeddings = self.drop1(embeddings)
        # embedding = self.fc1(embeddings)
        # embedding = F.relu(embedding)
        embedding = self.enhance(embedding)

        return embedding