import librosa
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import torch
"建立训练时的标签和语音类别的映射关系"

classes_labels = list({
    '5074': 689,
    '6045': 299,
    '6201': 487,
    '5093': 178,
    '6156': 261,
    '5096': 184,
    '2364': 360,
    '6014': 186,
    '0637': 31149,
    '0638': 33843,
    '9001': 1648,
    '9003': 2366,
    '9002': 1381
}.keys())

types = 'awake diaper hug hungry sleepy uncomfortable'.split()

def extract_features():
    """
    Extracting features form Spectrogram
    We will extract
        MFCC (20 in number)
        Spectral Centroid (光谱质心)
        Zero Crossing Rate (过零率)
        Chroma Frequencies (色度频率)
        Spectral Roll-off (光谱衰减)
    """
    header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    return header.split()


def write_data_to_csv_file(header, indexes, filename, selection):
    """
    Writing data to csv file
    Notation: we must close the file
    """
    file = open(filename, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        file.close()
        for key, value in indexes.items():
            wav, sample_rate = librosa.load("data_path", mono=True, duration=15)
            chroma_stft = librosa.feature.chroma_stft(y=wav, sr=sample_rate)
            rms = librosa.feature.rms(y=wav)
            spec_cent = librosa.feature.spectral_centroid(y=wav,
                                                          sr=sample_rate)
            spec_bw = librosa.feature.spectral_bandwidth(y=wav, sr=sample_rate)
            rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sample_rate)
            zcr = librosa.feature.zero_crossing_rate(y=wav)
            mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate)
            to_append = f'{(key if selection == "train" else os.path.split(key)[1])} {np.mean(chroma_stft)} ' \
                        f'{np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} '
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {value}'
            file = open(filename, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                file.close()


def write_result_to_csv(data_file, filename, results):
    """
    输出测试集合的结果到csv
    :param filename: 文件名称
    :param data_file: 测试集
    :param results: 识别结果
    :return:
    """
    file = open(filename, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow('id label'.split())
        file.close()
        data = pd.read_csv(data_file)
        wav_paths = data.iloc[:, [0]].T
        wav_paths = np.matrix.tolist(wav_paths)[0]

        dictionary = {}
        print(len(results))
        for i in range(len(results)):
            dictionary[wav_paths[i]] = results[i]
            to_append = f'{wav_paths[i]} {classes_labels[results[i].data.numpy()]} '
            file = open(filename, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                file.close()