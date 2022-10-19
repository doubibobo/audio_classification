from src.model.submodules.image_head.resnet import ResNet
from src.model.submodules.text_head.gru import GRUModel
from src.model.submodules.text_head.lstm import LSTMModel


def get_instance(name, parameters_dict):
    model = {
        'ResNet': ResNet,
        'GRU': GRUModel,
        'LSTM': LSTMModel
    }[name]
    return model(**parameters_dict)
