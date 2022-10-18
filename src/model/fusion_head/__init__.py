from src.model.fusion_head.transformer import BertEncoder, BertPooler
from src.model.common_head.se_net import SENet
from src.model.common_head.mean_pooler import MeanPooling
from src.model.fusion_head.concation_dense_se import ConcatDenseSE


def get_instance(name, parameters_dict):
    model = {
        'Transformer': BertEncoder,
        'Pooler': BertPooler,
        'SENet': SENet,
        'ConcatDenseSE': ConcatDenseSE,
        'MeanPooling': MeanPooling
    }[name]
    return model(**parameters_dict)
