import torch.nn
import torch.nn as nn
import torch.nn.functional as F

class LogisticModel(nn.Module):
    def __init__(self, classes_number=2, input_dim=512, l2_penalty=None, dropout_rate=0.1):
        super().__init__()
        self.classes_number = classes_number
        self.input_dim = input_dim
        self.l2_penalty = l2_penalty if l2_penalty is not None else 0.0
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.fc_init = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, classes_number)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_feature):
        logistics = F.relu(self.fc_init(input_feature))
        logistics = self.dropout_layer(logistics)
        logistics = self.fc(input_feature)
        # 针对分类问题
        output = self.softmax(logistics)
        return {
            'predictions': torch.max(output, dim=1)[1],
            'logistics': logistics,
            # 'fusion_features': input_feature.cpu().detach().numpy().tolist()
        }