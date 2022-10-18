import os
import shutil
import torch

from torch.utils.tensorboard import SummaryWriter


class Summary:
    def __init__(self, summary_root):
        if not os.path.exists(summary_root):
            os.makedirs(summary_root)
        else:
            shutil.rmtree(summary_root)
        self.summary_writer = SummaryWriter(summary_root)

    def summary_model(self, model, test_input_to_model):
        """
        保存模型整体结构
        :param model:
        :param test_input_to_model: 模型的理论输入
        :return:
        """
        model.eval()
        with torch.no_grad():
            self.summary_writer.add_graph(
                model=model, input_to_model=test_input_to_model, verbose=False
            )

    def summary_writer_add_scalars(self, global_step, train_message, valid_message):
        """
        同时记录多个指标
        :param global_step:
        :param train_message:
        :param valid_message:
        :return:
        """
        for key, value in train_message.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.summary_writer.add_scalars(
                        "compared/{}/{}".format(key, sub_key),
                        {
                            "train": train_message[key][sub_key],
                            "valid": valid_message[key][sub_key],
                        },
                        global_step,
                    )
            else:
                self.summary_writer.add_scalars(
                    "compared/{}".format(key),
                    {"train": train_message[key], "valid": valid_message[key]},
                    global_step,
                )

    def summary_writer_add_scalar(self, global_step, message, mean_tag):
        """
        仅仅记录一个指标
        :param global_step:
        :param message:
        :param mean_tag:
        :return:
        """
        for key, value in message.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.summary_writer.add_scalar(
                        "{}/{}/{}".format(mean_tag, key, sub_key),
                        message[key][sub_key],
                        global_step,
                    )
            else:
                self.summary_writer.add_scalar(
                    "{}/{}".format(mean_tag, key), message[key], global_step
                )

    def train_valid_compare_summary(
        self,
        train_losses,
        train_accuracies,
        valid_losses,
        valid_accuracies,
        count_valid,
    ):
        """
        记录迭代时的日志记录（包含验证集的评价）
        :param train_losses: list
        :param train_accuracies:list, sub item is dict, shape is same as the output of calculate accuracy
        :param valid_losses: list
        :param valid_accuracies: list, sub item is dict, shape is same as the output of calculate accuracy
        :param count_valid: int
        """
        train_message = self.summary_return(train_losses, train_accuracies)
        valid_message = self.summary_return(valid_losses, valid_accuracies)

        self.summary_writer_add_scalars(count_valid, train_message, valid_message)
        print(" * Train-Metrics are: {}.\n".format(str(train_message)))
        print(" * Valid-Metrics are: {}.\n".format(str(valid_message)))

    def train_summary(self, train_losses, train_accuracies, count_summary):
        """
        训练过程中的日志记录（不包含验证集的评价）
        :param train_losses:
        :param train_accuracies:
        :param count_summary:
        :return:
        """
        train_message = self.summary_return(train_losses, train_accuracies)
        self.summary_writer_add_scalar(count_summary, train_message, "train/")
        print("Train_messages are as follows: {}|n".format(**train_message))

    def learning_rate_summary(self, learning_rate_params, global_step):
        lrs = {}
        for param_group in learning_rate_params:
            group_id = len(lrs)
            lrs["{}".format(group_id)] = param_group["lr"]

        self.summary_writer.add_scalars(
            "lr", lrs, global_step,
        )
        pass

    @staticmethod
    def summary_return(loss, metrics):
        return {"loss": loss, **metrics}
