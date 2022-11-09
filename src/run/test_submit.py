import os
import sys

import torch
import torch.multiprocessing
import struct
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from collections import Counter

sys.path.append("/home/data/zhuchuanbo/Documents/competition/JHT")
print(sys.path)
from configs.config_for_optuna import Config
from configs.get_args import parse_args
from src.utils.get_data import get_data
from src.utils.train_tool import set_seed

os.environ['TORCH_HOME'] = '/home/data/zhuchuanbo/Documents/pretrained_models'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"  # This is crucial for reproducibility

DATASET_PATH = "/home/data/zhuchuanbo/Documents/competition/JHT/data"
DATA_SAVE_PATH = os.path.join(DATASET_PATH, "TestData")
SPECTROGRAM_PATH = os.path.join(DATASET_PATH, "Processed/test/spectrogram")
FIGURE_PATH = os.path.join(DATASET_PATH, "Processed/test/figures")
NPY_SAVE_PATH = "/home/data/zhuchuanbo/Documents/competition/JHT/data/Processed/test/test_data.npy"

label_to_class = {
    0: '5074',
    1: '6045',
    2: '6201',
    3: '5093',
    4: '6156',
    5: '5096',
    6: '2364',
    7: '6014',
    8: '0637',
    9: '0638',
    10: '9001',
    11: '9003',
    12: '9002',
    13: '9999'
}


# 模型保存路径
test_pretrained_path = "/home/data/zhuchuanbo/Documents/competition/JHT/checkpoints/JHT/JHTModel_2022-10-25_16_57/best_model_0.pth" 

email_files = "/home/data/zhuchuanbo/Documents/competition/JHT/src/run/HYSZ-华中科技大学-陈进才"

# 设置系列参数
global_args = Config(parse_args()).get_config()
set_seed(global_args.seed)

test_data = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_file_process():    
    def get_data_from_datfile(file_path):
        def plot_spectrogram(file_name, wav_data):
            """
            提取每条音频的频谱特征图
            """
            if os.path.exists(os.path.join(SPECTROGRAM_PATH, "{}.png".format(file_name))):
                return;
            cmap = plt.get_cmap('inferno')
            plt.figure(figsize=(10, 10))
            plt.specgram(wav_data,
                        NFFT=2048,
                        Fs=2,
                        Fc=0,
                        noverlap=128,
                        cmap=cmap,
                        sides='default',
                        mode='default',
                        scale='dB')
            plt.axis("off")
            plt.savefig(
                os.path.join(SPECTROGRAM_PATH, "{}.png".format(file_name)),
                bbox_inches='tight', pad_inches=0.0
            )
            plt.clf()
            
        data_file = open(str(file_path).split(".")[0] + ".dat", "rb")
        data_shorts = []
        while True:
            try:
                data_temp = data_file.read(2)
                data_short, = struct.unpack('h', data_temp)
                data_shorts.append(data_short)
            except Exception:
                print("Finish file reading!")
                break

        temp = open(str(file), "r", encoding='iso-8859-1')
        lines = temp.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].encode("iso-8859-1").decode('gbk').encode(
                'utf8').decode('utf8')
        ids = [
            int(lines[i].replace("\n", "").split("\t")[0])
            for i in range(5, len(lines))
        ]
        toas = [
            int(lines[i].replace("\n", "").split("\t")[1])
            for i in range(5, len(lines))
        ]
        pulse_start = [
            int(lines[i].replace("\n", "").split("\t")[2]) // 2
            for i in range(5, len(lines))
        ]
        pulse_length = [
            int(lines[i].replace("\n", "").split("\t")[3]) // 2
            for i in range(5, len(lines))
        ]

        for i in range(len(ids)):
            data_start = pulse_start[i]
            data_end = min(pulse_start[min(i + 1,
                                        len(pulse_start) - 1)],
                        len(data_shorts) - 1)
            data_temp = data_shorts[data_start:data_end]

            try:
                if len(data_temp) > 0:
                    if len(data_temp) >= 8072:
                        data_temp = data_temp[0: 8072]
                    else:
                        ii, data_temp_length = 0, len(data_temp)
                        while len(data_temp) < 8072:
                            data_temp.append(data_temp[ii])
                            ii = (ii + 1) % data_temp_length
                    
                    data_temp = data_temp / np.max(data_temp)
                    
                    pulse_data = {
                        "id": ids[i],
                        "toa": toas[i],
                        "data": data_temp,
                        "pulse_width": pulse_length[i],
                        "pulse_length": len(data_temp),
                        "file_name": '/'.join(file.parts[-3:]),
                        "spectrogram_file": "{}_{}".format(file.stem, str(i).zfill(5)),
                    }
                    # plot_spectrogram(pulse_data["spectrogram_file"], data_temp)
                    test_data.append(pulse_data)
                    
            except Exception:
                # print(wav)
                print("{}-{}".format(data_start, data_end))
                print(data_temp)

    files = Path("{}".format(DATA_SAVE_PATH))
    txt_list = list(files.glob("*.txt"))
    for file in txt_list:
        get_data_from_datfile(file)
    np.save(NPY_SAVE_PATH, test_data)


def get_test_accuracy():
    _, _, test_dataloader = get_data(global_args)
    model = torch.load(test_pretrained_path)["state_dict"]
    # model = model.to(device)
    model = torch.nn.parallel.DataParallel(model.to(device))
    model.eval()
    with torch.no_grad():
        test_predictions, test_paths = [], []
        test_logistics = []
        for _, (test_data, _, (test_path, _)) in enumerate(test_dataloader):
            for key, value in test_data.items():
                test_data[key] = value.to(device, non_blocking=True)
            model_output = model(test_data)
            test_predictions.extend(model_output["predictions"].cpu().numpy().tolist())
            model_output["logistics"] = F.softmax(model_output["logistics"], dim=1)
            test_logistics.extend(model_output["logistics"].cpu().numpy().tolist())
            test_paths.extend(test_path)
            # break
        print(len(test_predictions))
        print(len(test_logistics))
        print(len(test_paths))
        
        predictions = {}
        for file_temp, prediction_temp, test_logistic in zip(test_paths, test_predictions, test_logistics):
            key = "_".join(file_temp.split("/")[-1].split(".")[0].split("_")[0: -1])
            if key in predictions.keys():
                if test_logistic[np.argmax(test_logistic)] <= 0.85:
                    predictions[key].append(13)
                else:
                    predictions[key].append(np.argmax(test_logistic))
            else:
                if test_logistic[np.argmax(test_logistic)] <= 0.85:
                    predictions[key] = [13]
                else:
                    predictions[key] = [np.argmax(test_logistic)]
        
        for key, value in predictions.items():
            print("{}".format(Counter(value).most_common()))
            print("{}: {}: {}".format(key, label_to_class[stats.mode(value)[0][0]], stats.mode(value)[1][0] / len(value)))
            result_output = label_to_class[stats.mode(value)[0][0]]
            if stats.mode(value)[1][0] / len(value) >= 0.7:
                result_output = label_to_class[stats.mode(value)[0][0]]
            else:
                result_output = 9999
                
            file_name = os.path.join(email_files, "R_{}_{}.txt".format(key.split("_")[1], result_output))
            with open(file_name,'a') as file:
                pass
            
            
# test_file_process()
get_test_accuracy()