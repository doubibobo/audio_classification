import os
import time

import numpy as np
import optuna
import torch
from src.utils.calculate_metrics import calculate_metrics, print_confusion_matrix
from src.utils.get_summary import Summary


class TrainProcess:

    def __init__(
        self,
        args,
        train_loader,
        valid_loader,
        test_loader,
        model,
        model_root,
        criteria,
        optimizer,
        schedule,
        summary_root,
        device="cpu",
        cross_validation=0,
    ):
        self.args = args

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.model_root = model_root

        self.criterion = criteria
        self.optimizer = optimizer
        self.schedule = schedule
        self.cross_validation = cross_validation

        self.summary = Summary(summary_root)

        self.device = device

        self.model = torch.nn.parallel.DataParallel(model.to(device))

        self.epoch = 0
        self.best_epoch = 0
        self.best_accuracy = 0

    def train_process(self, max_epoch, wait_epoch, iteration, trail):
        start_time = time.time()
        train_step = iteration
        kb_param = []
        while True:
            print(" *** epoch {} is begin ...".format(train_step //
                                                      len(self.train_loader)))
            self.model.train()
            train_losses, train_predictions, train_labels = [], [], []

            # for train_data, train_label in tqdm(self.train_loader):
            for batch_index, (train_data, train_label, _) in enumerate(self.train_loader):
                for key, value in train_data.items():
                    train_data[key] = value.to(self.device)
                train_label = train_label.to(self.device)

                self.optimizer.zero_grad()
                model_output = self.model(train_data)
                loss = self.criterion(model_output["logistics"], train_label)
                loss.backward()

                # TODO 需要进一步确定阈值
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)

                self.optimizer.step()

                train_losses.append(loss.detach().cpu().numpy())
                train_predictions.extend(
                    model_output["predictions"].cpu().numpy().tolist())
                train_labels.extend(train_label.cpu().numpy().tolist())

                train_step += 1

            self.summary.learning_rate_summary(self.optimizer.param_groups,
                                               self.epoch)

            train_loss = np.mean(train_losses)
            train_metrics = calculate_metrics(
                train_predictions,
                train_labels,
                self.model.module.classifier.fc.out_features,
                dataset_name=self.args.dataset_name)
            # 进行验证集的验证
            valid_loss, valid_metrics = self.valid_process()
            self.summary.train_valid_compare_summary(train_loss, train_metrics,
                                                     valid_loss, valid_metrics,
                                                     self.epoch)
            # 保存模型：包含条件判断、模型保存等
            self.save_model(valid_metrics)

            print(" *** epoch {} is finished!".format(self.epoch))

            # 早停机制
            if self.epoch >= max_epoch or (self.epoch -
                                           self.best_epoch) >= wait_epoch:
                print(
                    " The epoch number is {}, and the best_epoch is {}, and the best accuracy is {}"
                    .format(self.epoch, self.best_epoch, self.best_accuracy))
                time_consuming = time.time() - start_time
                print(" Time consume for train process is {}s, about {}hours".
                      format(time_consuming, time_consuming / 3600))
                print("The kb param is : {}".format(kb_param))
                break

            if self.schedule is not None:
                self.schedule.step()

            trail.report(valid_metrics["accuracy"], self.epoch)

            # Handle pruning based on the intermediate value.
            if trail.should_prune():
                raise optuna.exceptions.TrialPruned()

            self.epoch += 1

    def valid_process(self):
        self.model.eval()
        with torch.no_grad():
            valid_losses, valid_predictions, valid_labels = [], [], []
            # for valid_data, valid_label in tqdm(self.valid_loader):
            for batch_index, (valid_data, valid_label,
                              _) in enumerate(self.valid_loader):
                for key, value in valid_data.items():
                    valid_data[key] = value.to(self.device, non_blocking=True)
                valid_label = valid_label.to(self.device, non_blocking=True)
                model_output = self.model(valid_data)
                loss = self.criterion(model_output["logistics"], valid_label)

                valid_losses.append(loss.cpu().numpy().tolist())
                valid_predictions.extend(
                    model_output["predictions"].cpu().numpy().tolist())
                valid_labels.extend(valid_label.cpu().numpy().tolist())

            valid_metrics = calculate_metrics(
                valid_predictions,
                valid_labels,
                self.model.module.classifier.fc.out_features,
                dataset_name=self.args.dataset_name)
            print("Confusion matrix for valid set:\n")
            print_confusion_matrix(valid_labels, valid_predictions,
                                   self.model.module.classifier.fc.out_features)

        return np.mean(valid_losses), valid_metrics

    def test_process(self, test_pretrained_path=None):
        # TODO 重大bug，测试时用的模型非最优模型
        if test_pretrained_path is not None:
            self.model = torch.load(test_pretrained_path)["state_dict"]
            self.model = self.model.to(self.device)
        else:
            self.model = torch.load(
                os.path.join(
                    self.model_root,
                    "best_model_{}.pth".format(self.cross_validation),
                ))["state_dict"]
            self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            test_losses, test_predictions, test_labels = [], [], []
            test_ids, text_origins, KB_origins = [], [], []
            features = []
            # for test_data, test_label in tqdm(self.test_loader):
            for batch_index, (test_data, test_label, _) in enumerate(self.test_loader):
                for key, value in test_data.items():
                    test_data[key] = value.to(self.device, non_blocking=True)
                test_label = test_label.to(self.device, non_blocking=True)

                model_output = self.model(test_data)
                loss = self.criterion(model_output["logistics"], test_label)

                test_losses.append(loss.cpu().numpy().tolist())
                test_predictions.extend(
                    model_output["predictions"].cpu().numpy().tolist())
                test_labels.extend(test_label.cpu().numpy().tolist())

            test_metrics = calculate_metrics(
                test_predictions,
                test_labels,
                self.model.classifier.fc.out_features,
                dataset_name=self.args.dataset_name)
            print("Confusion matrix for test set:\n")
            print_confusion_matrix(test_labels, test_predictions,
                                   self.model.classifier.fc.out_features)

            # if test_pretrained_path is not None:
            #     # 保存特征向量
            #     np.save(
            #         "visualization/" + self.args.dataset_name + "/mosi_test_fusion_features.npy",
            #         np.array(features))
            #     np.save(
            #         "visualization/" + self.args.dataset_name + "/mosi_test_tv_features.npy",
            #         np.array([feature[0:768] for feature in features]))
            #     np.save(
            #         "visualization/" + self.args.dataset_name + "/mosi_test_ta_features.npy",
            #         np.array([feature[768:768 * 2] for feature in features]))
            #     np.save(
            #         "visualization/" + self.args.dataset_name + "/mosi_test_kb_features.npy",
            #         np.array(
            #             [feature[768 * 2:768 * 3] for feature in features]))
            #     np.save(
            #         "visualization/" + self.args.dataset_name + "/mosi_test_t_features.npy",
            #         np.array(
            #             [feature[768 * 3:768 * 4] for feature in features]))

            #     # 保存标签
            #     np.save(
            #         "visualization/" + self.args.dataset_name + "/mosi_test_fusion_labels.npy",
            #         np.array(test_labels))

        return np.mean(test_losses), test_metrics

    def save_model(self, metrics):
        # TODO 将标准改为loss，因为优化的目标就是loss，accuracy只是一个表象
        if metrics["accuracy"] > self.best_accuracy:
            self.best_epoch = self.epoch
            self.best_accuracy = metrics["accuracy"]
            model_save_path = os.path.join(
                self.model_root,
                "best_model_{}.pth".format(self.cross_validation),
            )
            torch.save(
                {
                    "epoch":
                    self.epoch,
                    "metrics":
                    metrics,
                    "state_dict":
                    self.model.module,
                    "optimizer":
                    self.optimizer.state_dict(),
                    "scheduler":
                    self.schedule.state_dict()
                    if self.schedule is not None else 0,
                },
                model_save_path,
            )
