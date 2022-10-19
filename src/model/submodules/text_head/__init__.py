from src.model.submodules.text_head.bert import EmotionModel
from src.model.submodules.text_head.gru import GRUModel
from src.model.submodules.text_head.lstm import LSTMModel
from src.model.submodules.text_head.roBERTa import RobertaModelEmotion
from src.model.submodules.text_head.text_gcn import TextGCN

def get_instance(name, parameters_dict):
    model = {
        'Bert': EmotionModel,
        'Roberta': RobertaModelEmotion,
        'GRU': GRUModel,
        'LSTM': LSTMModel,
        'TextGCN': TextGCN
    }[name]
    return model(**parameters_dict)
