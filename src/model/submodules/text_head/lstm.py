import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        number_layers,
        drop_rate,
        batch_first=True,
        bidirectional=False,
    ):
        """
        :param input_size:
        :param hidden_size: the hidden_size is half of the input_size
        :param number_layers:
        :param drop_rate:
        :param batch_first:
        :param bidirectional:
        """
        super().__init__()

        if number_layers == 1:
            drop_rate = 0.0
            
        self.lstm_layer = nn.LSTM(
            input_size,
            hidden_size,
            number_layers,
            batch_first=batch_first,
            dropout=drop_rate,
            bidirectional=bidirectional,
        )
        # self.weigth_bias_init()
        self.fc_layer = nn.Linear((bidirectional + 1) * hidden_size, output_size)
        self.dropout_layer = nn.Dropout(p=drop_rate)

    def weigth_bias_init(self):
        # 采用正交初始化方法，
        # 参考：你在训练RNN的时候有哪些特殊的trick？ - YJango的回答 - 知乎 https://www.zhihu.com/question/57828011/answer/155275958
        for name, param in self.lstm_layer.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, text_features):
        """
        :param text_features: shape is <B, T, D> where T denotes the text length, D is the size of embedding shape
        :return:
        """
        # self.lstm_layer.flatten_parameters()  # 多卡训练设置
        lstm_output, output = self.lstm_layer(
            text_features
        )  # output shape is <B, T, 2 * hidden_size>
        output = output[0].squeeze()
        output = self.fc_layer(output)
        output = F.relu(output)
        output = self.dropout_layer(output)
        return {
            "embedding": output,
            "sequence": lstm_output,
        }
