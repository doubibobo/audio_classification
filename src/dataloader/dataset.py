from abc import ABC
from PIL import Image
from pathlib import Path

from torch.utils.data.dataset import Dataset
import importlib
import torch.nn.functional as F
import numpy as np
from random import sample

import os
import torch
from torchvision import transforms

class_to_label = {
    '5074': 0,
    '6045': 1,
    '6201': 2,
    '5093': 3,
    '6156': 4,
    '5096': 5,
    '2364': 6,
    '6014': 7,
    '0637': 8,
    '0638': 9,
    '9001': 10,
    '9003': 11,
    '9002': 12
}

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
    12: '9002'
}


def get_class_from_name(name):
    args = name.split(".")
    package_name = ""
    preprocess_class_name = args[-1]
    for i in range(len(args) - 1):
        package_name += args[i] + "."
    package_name = package_name[0:-1]

    preprocess_module = importlib.import_module(package_name)
    preprocess_class = getattr(preprocess_module, preprocess_class_name)
    return preprocess_class


class JHTDataset(Dataset, ABC):

    def __init__(
        self,
        image_dir,
        classes=13,
        slices=None,
        test_mode=False
    ):
        super(JHTDataset, self).__init__()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.image_dir = image_dir
        self.number_classes = classes

        self.image_label = []
        self.length = 0
        self.test_mode = test_mode

        self.__create_iamge_label_dict()
        
        # self.image_label = self.image_label[: 30000]
        # self.length = len(self.image_label)
        
        if slices is not None:
            data = np.array(self.image_label)
            self.image_label = data[slices]
            self.length = len(slices)
            
            
    def __create_iamge_label_dict(self):
        image_files = Path("{}".format(self.image_dir))
        if self.test_mode:
            files_list = list(image_files.glob("*/*.png"))
        else:
            files_list = []
            for dir in list(image_files.glob("*")):
                sub_files = list(image_files.glob("{}/*.png".format(dir.stem)))
                if dir.stem not in ["0637", "0638"]:
                    files_list.extend(sub_files)
                else:
                    files_list.extend(sample(sub_files, 3000))       
        for image_file in files_list:
            self.image_label.append(
                {
                    "path": str(image_file),
                    # "path": Image.open(os.path.join(image_file)).convert("RGB"),
                    "label": class_to_label[image_file.parts[-2]] if not self.test_mode else 0,
                }
            )
            self.length += 1

    def __getitem__(self, item):
        data = self.image_label[item]
        input_image = Image.open(os.path.join(data["path"])).convert("RGB")
        return (
            {
                "image_data": self.transform(input_image)
                # "image_data": data["path"],
                # "image_path": data["path"],
            },
            F.one_hot(
                torch.tensor(data["label"]),
                num_classes=self.number_classes
            ), 
            (
                data["path"], 
                label_to_class[data["label"]]
            )
        )

    def __len__(self):
        return self.length

    def get_all_data(self):
        data, label = [], []
        for item in range(len(self.image_label)):
            data.append(self.image_label[item]['path'])
            label.append(self.image_label[item]['label'])
        return np.array(data), np.array(label)


dataset = JHTDataset(
    image_dir=
    "/home/data/zhuchuanbo/Documents/competition/JHT/data/Processed/spectrogram",
    classes=13)
