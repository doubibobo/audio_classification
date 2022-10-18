import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="JHT",
        help="support JHT",
    )
    parser.add_argument(
        "--category_number", type=int, default=13, help="the categories of the input data."
    )
    parser.add_argument(
        "--pretrained_arch", type=str, default="resnet18", help="text pretrained model."
    )
    parser.add_argument(
        "--model_save_path", type=str, default="checkpoints", help="path to save model."
    )
    parser.add_argument(
        "--log_save_path", type=str, default="logs", help="path to save log"
    )
    parser.add_argument(
        "--res_save_path",
        type=str,
        default="results",
        help="path to save model's performance.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="num workers of loading data"
    )
    parser.add_argument(
        '--prefetch', 
        default=256, 
        type=int, 
        help="use for training duration per worker"
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/home/data/zhuchuanbo/Documents/competition/JHT",
        help="path of working directory",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="pretrained_models",
    )
    parser.add_argument(
        "--test_pretrained_path",
        type=str,
        default=None,
        help="default mode for train or test",
    )
    return parser.parse_args()
