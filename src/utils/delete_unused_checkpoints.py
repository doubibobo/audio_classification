import csv
import glob
import os
import shutil

# work_dir = '/'.join(os.getcwd().split('/')[:-2])
work_dir = "/".join(os.getcwd().split("/")[:])
checkpoints_dir = os.path.join(work_dir, "checkpoints")
logs_dir = os.path.join(work_dir, "logs")
results_dir = os.path.join(work_dir, "results")


def delete_checkpoints(dirs):
    """删除结果较差的训练结果
    """
    for dir in dirs:
        shutil.rmtree(dir)
        print("checkpoints dir {} is deleted!".format(dir))


def delete_log(dirs):
    """删除结果交叉的训练日志
    """
    for dir in dirs:
        shutil.rmtree(dir)
        print("log_dir {} is deleted!".format(dir))


def descending_sort(data, location):
    """降序排序

    Args:
        data (_type_): 待排序数据
        location (_type_): 基准元素位置
    """

    def get_base_item(item):
        """获取基准元素

        Args:
            item (_type_): 元素

        Returns:
            _type_: _description_
        """
        return item[location]

    data.sort(key=get_base_item, reverse=True)
    return data


def find_and_delete_files(
    dataset_name,
    model_name,
    data_flag="",
    reserved_number=5,
    default_name="performance_sum.csv",
):
    """寻找并删除不需要的模型

    Args:
        dataset_name (_type_): 数据集
        model_name (_type_): 模型名称
        data_flag (str, optional): 是否是带有时间戳的子目录. Defaults to ''.
        reserved_number (int, optional): 保存的最优结果数量. Defaults to 5.
        default_name (str, optional): 默认的总结文件名. Defaults to 'performance_sum.csv.csv'.
    """
    result_sum_path = os.path.join(
        results_dir, dataset_name, model_name, data_flag, default_name
    )
    csv_reader = csv.reader(open(result_sum_path))
    data = [line for line in csv_reader][:]
    print(len(data))

    data_list = [data[0]]
    data_list.extend(descending_sort(data[1:], 3)) # 2 or 1 or 3
    print(len(data_list))

    saved_checkpoints_list, saved_log_list = [], []

    for item in data_list[1 : (reserved_number + 1)]:
        saved_checkpoints_list.append(item[14]) # 18/19, 12/13
        saved_log_list.append(item[15])

    model_checkpoints_list = glob.glob(
        "{}/{}_*".format(os.path.join(checkpoints_dir, dataset_name), model_name)
    )[:]
    model_log_list = glob.glob(
        "{}/{}_*".format(os.path.join(logs_dir, dataset_name), model_name)
    )[:]

    deleted_checkpoints_list, deleted_log_list = [], []

    for item in model_checkpoints_list:
        full_path = item
        if full_path not in saved_checkpoints_list:
            deleted_checkpoints_list.append(full_path)
        else:
            pass

    for item in model_log_list:
        full_path = item
        if full_path not in saved_log_list:
            deleted_log_list.append(full_path)
        else:
            pass
    
    print("aligned" in deleted_checkpoints_list)
    print("aligned" in deleted_log_list)
    print("{}-{}".format(len(deleted_log_list), len(saved_log_list)))
    print("{}-{}".format(len(deleted_checkpoints_list), len(saved_checkpoints_list)))

    return deleted_checkpoints_list, deleted_log_list


list1, list2 = find_and_delete_files(
    dataset_name="JHT",
    model_name="JHTModel",
    data_flag="",
    reserved_number=10,
    default_name="resnet18_part_samples.csv"
)

delete_checkpoints(list1)
delete_log(list2)
