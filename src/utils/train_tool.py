import importlib
import os
import random

import dgl
import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    """
    设置随机数种子 for numpy and torch
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_class_from_name(name):
    args = name.split('.')
    package_name = ''
    preprocess_class_name = args[-1]
    for i in range(len(args) - 1):
        package_name += args[i] + '.'
    package_name = package_name[0: -1]

    preprocess_module = importlib.import_module(package_name)
    preprocess_class = getattr(preprocess_module, preprocess_class_name)
    return preprocess_class


def check_and_create_dir(args, path, time_dir='', subdir=True):
    """
    创建目录及子目录, subdir为Ture时, 默认创建子目录
    """
    if not os.path.exists(path):
        os.makedirs(path)
    if not subdir:
        return path
    path = os.path.join(path,
                        args.dataset_name,
                        '{0}{1}{2}'.format(args.net_type.split('.')[-1], '_' if time_dir != '' else '', time_dir))
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def deep_select_dict(dict1, prefix='', delimiter='_'):
    results = {}
    for key, value in dict1.items():
        if isinstance(value, dict):
            results = {**results, **deep_select_dict(value, key)}
        elif isinstance(value, torch.Tensor):
            results[prefix + delimiter + key] = value.cpu().tolist()
        else:
            results[prefix + delimiter + key] = value
    return results

def save_results(args, classification_result, confusion_array):
    """
    :param args: 模型参数
    :param classification_result: 分类结果
    :param confusion_array: 混淆矩阵结果
    :return:
    """
    # 保存confusion 矩阵
    confusion_save_path = os.path.join(args.model_save_path, 'confusion_matrix.txt')
    np.savetxt(confusion_save_path, confusion_array)

    # 保存模型参数——两份
    args_name = deep_select_dict(args)
    results = deep_select_dict(classification_result)

    args_save_path = os.path.join(args.model_save_path, 'args_file.txt')
    with open(args_save_path, 'a+') as f:
        print(args_name, file=f)
    result_save_path = os.path.join(args.model_save_path,
                                    args.dataset_name + '-' + args.modelName + '.csv')
    df = pd.DataFrame(columns=list(results.keys()) + list(args_name.keys()))

    df.loc[len(df)] = list(results.values()) + list(args_name.values())
    df.to_csv(result_save_path, index=False)
    print('Results are saved to %s...' % result_save_path)
    return {
        'args': args_name,
        'result': results,
        'result_save_path': result_save_path,
        'args_path': args_save_path
    }
