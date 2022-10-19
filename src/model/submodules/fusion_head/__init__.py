from src.model.submodules.fusion_head.transformer import BertEncoder, BertPooler
from src.model.submodules.common_head.se_net import SENet
from src.model.submodules.common_head.mean_pooler import MeanPooling
from src.model.submodules.fusion_head.concation_dense_se import ConcatDenseSE


def get_instance(name, parameters_dict):
    model = {
        'Transformer': BertEncoder,
        'Pooler': BertPooler,
        'SENet': SENet,
        'ConcatDenseSE': ConcatDenseSE,
        'MeanPooling': MeanPooling
    }[name]
    return model(**parameters_dict)
