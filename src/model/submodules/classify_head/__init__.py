from src.model.submodules.classify_head.logistic_model import LogisticModel


def get_instance(name, parameters_dict):
    model = {
        'LogisticModel': LogisticModel
    }[name]
    return model(**parameters_dict)
