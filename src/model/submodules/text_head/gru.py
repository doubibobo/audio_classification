import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        number_layers,
        drop_rate,
        batch_first=True,
        bidirectional=True,
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
            drop_rate = 0.
            
        self.gru_layer = nn.GRU(
            input_size,
            hidden_size,
            number_layers,
            batch_first=batch_first,
            dropout=drop_rate,
            bidirectional=bidirectional,
        )
        self.weigth_bias_init()

        # The normalized shape is <T * 2 * hidden_size> 仅对最后一个维度进行归一化
        self.norm_layer = nn.LayerNorm((bidirectional + 1) * hidden_size)
        self.activation_layer = nn.ReLU()
        self.fc_layer = nn.Linear((bidirectional + 1) * hidden_size, output_size)
        self.dropout_layer = nn.Dropout(p=drop_rate)

    def weigth_bias_init(self):
        # 采用正交初始化方法，
        # 参考：你在训练RNN的时候有哪些特殊的trick？ - YJango的回答 - 知乎 https://www.zhihu.com/question/57828011/answer/155275958
        for name, param in self.gru_layer.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, text_features):
        """
        :param text_features: shape is <B, T, D> where T denotes the text length, D is the size of embedding shape
        :return:
        """
        self.gru_layer.flatten_parameters()  # 多卡训练设置
        gru_output, h_n = self.gru_layer(text_features)  # output shape is <B, T, 2 * hidden_size>
        # output = text_features + gru_output
        # gru_output = self.activation_layer(gru_output)
        # output = gru_output
        output = h_n[-1]
        
        # output = self.norm_layer(output)
        # output = self.activation_layer(output)
        output = self.dropout_layer(output)
        output = self.fc_layer(output)
        # output = self.activation_layer(output)

        # # Only one modality
        # output = output[:, -1, :]
        
        return output


if __name__ == "__main__":
    data = torch.randn((32, 128, 768))
    model = GRUModel(768, 384, 768, 2, 0.1)
    ret = model(data)
    print(ret.shape)
