import struct
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

DATASET_PATH = "/home/data/zhuchuanbo/Documents/competition/JHT/data"
DATA_SAVE_PATH = "/home/data/zhuchuanbo/Documents/competition/JHT/data/Processed/data"
FIGURE_PATH = "/home/data/zhuchuanbo/Documents/competition/JHT/data/Processed/figures"
SPECTROGRAM_PATH = "/home/data/zhuchuanbo/Documents/competition/JHT/data/Processed/spectrogram"

# 存储样本信息
data_message = {}
# 存储样本信息
data = []
# 脉冲数量
pulse_number = []
# 信号载频 RF
rfs = []
# 采样率
fss = []
# 时间长度
time_length = []
# 平均脉冲时间
pulse_time_length = []


def func1():
    files = Path("{}".format(DATASET_PATH))
    file_list = list(files.glob("*/Train/*.dat"))
    for file in file_list:
        splits = file.stem.split("_")
        if splits[1] not in data_message.keys():
            data_message[splits[1]] = int(splits[3])
        else:
            data_message[splits[1]] += int(splits[3])
        pulse_number.append(int(splits[3]))
    print("脉冲总数: {}".format(sum(pulse_number)))
    print(data_message)


def func2():
    files = Path("{}".format(DATASET_PATH))
    file_list = list(files.glob("*/Train/*.txt"))
    for file in file_list:
        temp = open(str(file), "r", encoding='iso-8859-1')
        lines = temp.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].encode("iso-8859-1").decode('gbk').encode(
                'utf8').decode('utf8')
        rf = int("".join(list(filter(str.isdigit, lines[0]))))
        fs = int("".join(list(filter(str.isdigit, lines[1]))))
        rfs.append(rf)
        fss.append(fs)
        # 信号长度（按照秒来计算）
        time_length.append(
            int(lines[-1].split("\t")[1]) /
            (int(lines[1].split("：")[1]) * 1000000))
    print("信号载频种类数: {}".format(len(set(rfs))))
    print("最大RF: {}; 最小RF: {}".format(np.max(rfs), np.min(rfs)))
    print("采样频率种类数: {}".format(len(set(fss))))
    print("97\%分位: {}".format(
        np.sort(time_length)[int(len(time_length) * 0.97)]))
    print("96\%分位: {}".format(
        np.sort(time_length)[int(len(time_length) * 0.96)]))
    print("95\%分位: {}".format(
        np.sort(time_length)[int(len(time_length) * 0.95)]))
    print("最长时间: {}; 平均时间: {}; 最短时间: {}".format(np.max(time_length),
                                                np.mean(time_length),
                                                np.min(time_length)))


def func3():
    files = Path("{}".format(DATASET_PATH))
    txt_list = list(files.glob("*/Train/*.txt"))
    for file in txt_list:
        data_file = open(str(file).split(".")[0] + ".dat", "rb")
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
            pulse_time_length.append(len(data_temp))

            try:
                # wav, sample_rate = np.array(data_temp).astype(np.float32), 1
                # chroma_stft = librosa.feature.chroma_stft(y=wav,
                #                                           sr=sample_rate)
                # rms = librosa.feature.rms(y=wav)
                # spec_cent = librosa.feature.spectral_centroid(y=wav,
                #                                               sr=sample_rate)
                # spec_bw = librosa.feature.spectral_bandwidth(y=wav,
                #                                              sr=sample_rate)
                # rolloff = librosa.feature.spectral_rolloff(y=wav,
                #                                            sr=sample_rate)
                # mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate)
                # zcr = librosa.feature.zero_crossing_rate(y=wav)
                # mfcc_features = [
                #     np.mean(chroma_stft),
                #     np.mean(rms),
                #     np.mean(spec_cent),
                #     np.mean(spec_bw),
                #     np.mean(rolloff),
                #     np.mean(zcr)
                # ]
                # for e in mfcc:
                #     mfcc_features.append(np.mean(e))
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
                        "pulse_label": file.stem.split("_")[1],
                        # "mfcc_features": mfcc_features,
                        "file_name": '/'.join(file.parts[-3:]),
                        "spectrogram_file": "{}_{}".format(file.stem, str(i).zfill(5)),
                    }
                    data.append(pulse_data)
                    # func5(pulse_data["pulse_label"], pulse_data["spectrogram_file"], data_temp)
                    func6(
                        pulse_data["data"], 
                        "{}/{}_{}.png".format(FIGURE_PATH, '_'.join(file.parts[-3:]).replace(".txt", ""), str(i))
                    )

            except Exception:
                # print(wav)
                print("{}-{}".format(data_start, data_end))
                print(data_temp)

            # if len(data_temp) >= 7500:


    np.save(DATA_SAVE_PATH, data)
    print(len(data))
    print("脉冲总数: {}".format(sum(pulse_number)))


def func4():
    print("单条脉冲信息")
    print(sorted(pulse_time_length))
    print("最长时间: {}; 平均时间: {}; 最短时间: {}".format(np.max(pulse_time_length),
                                                np.mean(pulse_time_length),
                                                np.min(pulse_time_length)))
    print("97\%分位: {}".format(
        sorted(pulse_time_length)[int(len(pulse_time_length) * 0.97)]))
    print("96\%分位: {}".format(
        sorted(pulse_time_length)[int(len(pulse_time_length) * 0.96)]))
    print("95\%分位: {}".format(
        sorted(pulse_time_length)[int(len(pulse_time_length) * 0.95)]))


def func5(label_type, file_name, wav_data):
    """
    提取每条音频的频谱特征图
    """
    cmap = plt.get_cmap('inferno')
    plt.figure(figsize=(10, 10))
    # 创建对应类的频谱图文件夹
    if not os.path.exists(os.path.join(SPECTROGRAM_PATH, label_type)):
        os.makedirs(os.path.join(SPECTROGRAM_PATH, label_type))

    # 加载15s的音频，转换为单声道（mono）
    # wav, sample_rate = librosa.load("audio_path", mono=True, duration=15)
    
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
        os.path.join(SPECTROGRAM_PATH, label_type, "{}.png".format(file_name)),
        bbox_inches='tight', pad_inches=0.0
    )
    plt.clf()

def func6(data_temp, file_name):
    """画出数据的波形图
    """
    time = np.arange(0, len(data_temp)) * (1.0)

    plt.plot(time, data_temp)
    plt.title("Wavform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.savefig(file_name, dpi=600)
    plt.clf()

# func1()
# func2()
func3()
func4()
