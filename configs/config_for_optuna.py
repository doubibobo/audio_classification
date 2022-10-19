import os

from configs.storge import Storage


class Config:

    def __init__(self, args):
        self.work_dir = args.work_dir
        self.pretrained_models_dir = args.pretrained_path
        self.dataset_name = args.dataset_name.upper()
        self.category_number = args.category_number
        self.pretrained_arch = args.pretrained_arch

        try:
            self.global_params = vars(args)
        except TypeError:
            self.global_params = args

    def __dataset_common_params(self):
        return {
            "JHT": {
                "image_dir": os.path.join(self.work_dir, "data/Processed/spectrogram"),
            }
        }

    def __model_config_params(self, trial=None):
        if trial is not None:
            visual_lr = trial.suggest_float("visual_lr", 1e-4, 1e-3, step=1e-4)
            visual_wd = trial.suggest_float("visual_wd", 1e-3, 1e-2, step=1e-3)
            other_lr = trial.suggest_float("other_lr", 1e-3, 1e-2, step=1e-3)
            other_wd = trial.suggest_float("other_wd", 1e-3, 5e-3, step=1e-3)
        else:
            visual_lr, visual_wd = 0.1, 0.1
            other_lr, other_wd = 0.1, 0.1

        batch_size = 256
        feature_droprate = 0.1
        optimizer = "torch.optim.Adam"

        schedule_type = 2

        return {
            "common_params": {
                "train_type": "src.utils.train.TrainProcess",
                "net_type": "src.model.models.resnet.JHTModel",
                "model_config": {
                    "pretrained_arch": "resnet50",
                    "pretrained_path": self.pretrained_models_dir,
                    "classifier_type": "LogisticModel",
                    "classifier_params": {
                        "classes_number": self.category_number,
                        "input_dim": 1024 * 1 * 1,
                        "l2_penalty": 0.0,
                        "dropout_rate": feature_droprate * 1,
                    },
                },
                "learning_rate": {
                    "visual": visual_lr,
                    "other": other_lr,
                },
                "weight_decay": {
                    "visual": visual_wd,
                    "other": other_wd,
                },
                "batch_size": batch_size,
                "early_stop": 5,
                "max_epoch": 10,
                "optimizer": optimizer,
                "scheduler": [
                    {
                        "name": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                        "params": {
                            "T_0": 2,
                            "T_mult": 2
                        },
                    },
                    {
                        "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
                        "params": {
                            "T_max": 32,
                            "eta_min": 0
                        },
                    },
                    {
                        "name": "torch.optim.lr_scheduler.ExponentialLR",
                        "params": {
                            "gamma": 0.95,
                            "last_epoch": -1
                        },
                    },
                ][schedule_type],
            },
            "dataset_params": {
                "JHT": {
                    "class_number": 13,
                    "criterion": {
                        # "name": "src.utils.loss.CrossEntropyLoss",
                        "name": "src.utils.loss.ASLSingleLabel",
                        "init_params": {
                            "reduction": "mean",
                        },
                    },
                    "dataset_class": "src.dataloader.dataset.JHTDataset",
                },
            },
        }

    def get_config(self, trial=None):
        return Storage(
            dict(
                self.global_params,
                **self.__model_config_params(trial)["dataset_params"][self.global_params["dataset_name"]],
                **self.__model_config_params(trial)["common_params"],
                **self.__dataset_common_params()[self.global_params["dataset_name"]]))
