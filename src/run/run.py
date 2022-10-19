import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
import torch
import torch.multiprocessing
from optuna.trial import TrialState

os.environ['TORCH_HOME'] = '/home/data/zhuchuanbo/Documents/pretrained_models'
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ[
    "CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"  # This is crucial for reproducibility

sys.path.append("/home/data/zhuchuanbo/Documents/competition/JHT")
print(sys.path)

from configs.config_for_optuna import Config
from configs.get_args import parse_args
from src.utils.get_data import get_data
from src.utils.train_tool import (
    check_and_create_dir,
    deep_select_dict,
    get_class_from_name,
    set_seed,
)

experiment_results = []
experiment_columns = []
experiment_save_path = ""


def objective(trail):
    # 设置系列参数
    global_args = Config(parse_args()).get_config(trail)
    global_args.log_save_path = os.path.join(global_args.work_dir,
                                             global_args.log_save_path)
    global_args.model_save_path = os.path.join(global_args.work_dir,
                                               global_args.model_save_path)
    global_args.res_save_path = os.path.join(global_args.work_dir,
                                             global_args.res_save_path)

    time_dir = time.strftime("%Y-%m-%d_%H_%M")
    global_args.log_save_path = check_and_create_dir(global_args,
                                                     global_args.log_save_path,
                                                     time_dir)
    global_args.model_save_path = check_and_create_dir(
        global_args, global_args.model_save_path, time_dir)
    global_args.res_save_path = check_and_create_dir(global_args,
                                                     global_args.res_save_path)
    print(global_args)

    set_seed(global_args.seed)
    test_losses, test_metrics = do_train(global_args, trail)
    print("\n The test loss is: {} \n, and the test metrics are: {}\n".format(
        test_losses, test_metrics))

    # 将结果保存至前面提到的文件中
    if len(experiment_results) == 0:
        experiment_columns.extend(["id"])
        experiment_columns.extend(["loss"])
        experiment_columns.extend(list(deep_select_dict(test_metrics).keys()))
        experiment_columns.extend(list(deep_select_dict(global_args).keys()))
        experiment_columns.extend(["args_save_path"])
    else:
        pass
    experiment_result = [len(experiment_results) + 1]
    experiment_result.extend([test_losses])
    experiment_result.extend(list(deep_select_dict(test_metrics).values()))
    experiment_result.extend(list(deep_select_dict(global_args).values()))
    experiment_result.extend([global_args.model_save_path])
    experiment_results.append(experiment_result)

    # 将参数暂时保存到模型保存中
    with open(os.path.join(global_args.model_save_path, "args_file.txt"),
              "a+") as file:
        print(json.dumps(global_args), file=file)

    # 保存当前结果
    item_df = pd.DataFrame(columns=experiment_columns,
                           data=[experiment_result])
    item_df.to_csv(
        os.path.join(
            global_args.res_save_path,
            "performance_{}.csv".format(len(experiment_results)),
        ),
        index=False,
    )
    print(" Experiment results are save to {}.\n".format(
        os.path.join(
            global_args.res_save_path,
            "performance_{}".format(len(experiment_results)),
        )))

    global experiment_save_path
    experiment_save_path = global_args.res_save_path

    return test_metrics["accuracy"]


def do_train(args, trail):
    using_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if using_cuda else "cpu")
    train_dataloader, valid_dataloader, test_dataloader = get_data(args)

    test_losses = [None for _ in range(len(train_dataloader))]
    test_accuraies = [None for _ in range(len(train_dataloader))]

    for i in range(len(train_dataloader)):
        "K折交叉验证的结果"
        model = get_class_from_name(args.net_type)(args.model_config)
        criterion = get_class_from_name(args.criterion["name"])(**args.criterion["init_params"]).to(device)

        visual_params = [
            p for n, p in list(model.image_head.named_parameters())
        ]
        other_params = [
            p for n, p in list(model.named_parameters())
            if 'image_head' not in n
        ]

        optimizer = get_class_from_name(args.optimizer)([
            {
                "params": visual_params,
                "weight_decay": args.weight_decay["visual"],
                "lr": args.learning_rate["visual"],
            },
            {
                "params": other_params,
                'weight_decay': args.weight_decay["other"],
                "lr": args.learning_rate["other"],
            },
        ], )

        scheduler = get_class_from_name(args.scheduler["name"])(
            optimizer=optimizer, **args.scheduler["params"])
        train_process = get_class_from_name(args.train_type)(
            args,
            train_dataloader[i],
            valid_dataloader[i],
            valid_dataloader[i],
            # test_dataloader,
            model,
            args.model_save_path,
            criterion,
            optimizer,
            scheduler,
            args.log_save_path,
            device=device,
            cross_validation=i,
        )

        if args.test_pretrained_path is None:
            train_process.train_process(
                max_epoch=args.max_epoch,
                wait_epoch=args.early_stop,
                iteration=0,
                trail=trail,
            )
        test_losses[i], test_metrics = train_process.test_process(
            args.test_pretrained_path)
        test_accuraies[i] = test_metrics["accuracy"]
    test_metrics["accuracy"] = np.mean(test_accuraies)
    test_losses = np.mean(test_losses)
    return test_losses, test_metrics


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="jht")
    study.optimize(objective, n_trials=20)
    pruned_trials = study.get_trials(deepcopy=False,
                                     states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False,
                                       states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # 保存所有实验参数
    df = pd.DataFrame(columns=experiment_columns, data=experiment_results)
    df.to_csv(os.path.join(experiment_save_path, "performance_sum.csv"),
              index=False)
