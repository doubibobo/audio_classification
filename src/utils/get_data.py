import torch
from src.utils.train_tool import get_class_from_name

from functools import partial
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, KFold


def get_data(args):
    data_loader_params = {
        "image_dir": args.image_dir,
        "classes": args.class_number,
    }

    dataset = get_class_from_name(args.dataset_class)(**data_loader_params)
    data, label = dataset.get_all_data()
    # # 层次划分数据集
    # split = StratifiedShuffleSplit(n_splits=5,
    #                                test_size=0.2,
    #                                random_state=args.seed)
    # # 随机采样
    # split = ShuffleSplit(
    #     n_splits=args.kfolds,
    #     test_size=args.val_ratio,
    #     random_state=args.seed
    # )
    split = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_loaders, valid_loaders = [], []
    for train_index, valid_index in split.split(data, label):
        print("TRAIN length: {}; VALID length: {}".format(
            len(train_index), len(valid_index)))
        print("TRAIN: ", train_index, "TEST:", valid_index)

        data_loader_params["slices"] = sorted(train_index)
        data_loader_params['test_mode'] = False
        train_dataset = get_class_from_name(
            args.dataset_class)(**data_loader_params)

        data_loader_params["slices"] = sorted(valid_index)
        data_loader_params['test_mode'] = False
        val_dataset = get_class_from_name(
            args.dataset_class)(**data_loader_params)

        if args.num_workers > 0:
            dataloader_class = partial(DataLoader,
                                       pin_memory=True,
                                       num_workers=args.num_workers,
                                       prefetch_factor=args.prefetch)
        else:
            # single-thread reading does not support prefetch_factor arg
            dataloader_class = partial(DataLoader,
                                       pin_memory=True,
                                       num_workers=0)

        train_dataloader = dataloader_class(
            train_dataset,
            batch_size=args.batch_size,
            sampler=RandomSampler(train_dataset),
            drop_last=True)
        val_dataloader = dataloader_class(
            val_dataset,
            batch_size=args.batch_size,
            sampler=SequentialSampler(val_dataset),
            drop_last=False)

        train_loaders.append(train_dataloader)
        valid_loaders.append(val_dataloader)

        data_loader_params['slices'] = None
        data_loader_params['test_mode'] = True
        dataset = get_class_from_name(args.dataset_class)(**data_loader_params)
        sampler = SequentialSampler(dataset)
        test_dataloader = DataLoader(dataset,
                                     batch_size=args.batch_size * 2,
                                     sampler=sampler,
                                     drop_last=False,
                                     pin_memory=True,
                                     num_workers=args.num_workers,
                                     prefetch_factor=args.prefetch)

        print(" The length of train data is {}".format(len(train_dataloader)))
        print(" The length of valid data is {}".format(len(val_dataloader)))
        print(" The length of test data is {}".format(len(test_dataloader)))
    return train_loaders, valid_loaders, test_dataloader

    # train_dataloader = torch.utils.data.DataLoader(
    #     get_class_from_name(args.dataset_class)(**data_loader_params),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=False,
    # )

    # data_loader_params["arch"] = "valid"
    # valid_dataloader = torch.utils.data.DataLoader(
    #     get_class_from_name(args.dataset_class)(**data_loader_params),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=False,
    # )

    # data_loader_params["arch"] = "test"
    # test_dataloader = torch.utils.data.DataLoader(
    #     get_class_from_name(args.dataset_class)(**data_loader_params),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=False,
    # )

    # return train_dataloader, valid_dataloader, test_dataloader
