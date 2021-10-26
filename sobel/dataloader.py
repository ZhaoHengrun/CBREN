import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
import numpy as np
import h5py
from torch import nn
import cv2
import torch
import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def load_img(filepath):
    img = Image.open(filepath)
    img_array = np.array(img)
    # img_tensor = torch.from_numpy(img_array)
    return img_array

def transform():
    return Compose([
        #CenterCrop(128),
        ToTensor()
    ])

# def generate_label_map(x):
#     label = np.zeros((x.shape[0], x.shape[1],2))
#     for i in range(label.shape[0]):
#         for j in range(label.shape[1]):
#             label[i,j,0] = i/label.shape[0]-0.5
#             label[i,j,1] = j/label.shape[1]-0.5
#     label = label.astype(np.float32)
#     return np.concatenate((x,label),axis=(-1))

# def generate_std_label_map():
#     label = np.zeros((2, 1024, 1536))
#     for i in range(label.shape[1]):
#         for j in range(label.shape[2]):
#             label[0,i,j] = i/label.shape[1]-0.5
#             label[1,i,j] = j/label.shape[2]-0.5
#     return label
#
# def crop_std(x):
#     #print(x.shape)
#     return x[0:1024,0:1536]


def get_training_set():
    root_dir = 'data/Training/'
    train_dir = join(root_dir, "data_256_h5")
    # print(train_dir)

    return DatasetFromFolder(train_dir, input_transform=transform(), target_transform=transform())

def get_test_set():
    root_dir = 'data/val_modcrop/'
    val_dir = join(root_dir, "data_h5")

    return DatasetFromFolder(val_dir, input_transform=transform(), target_transform=transform())

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir_1, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir_1, x) for x in listdir(image_dir_1)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        h5_data = h5py.File(self.image_filenames[index],'r')

        input = np.array(h5_data['input'])
        target = np.array(h5_data['target'])

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        # print(target.shape)
        target_scale2 = F.interpolate(target.unsqueeze(0),(128,128),mode='bicubic', align_corners=True).squeeze(0)
        target_scale3 = F.interpolate(target.unsqueeze(0),(64,64),mode='bicubic', align_corners=True).squeeze(0)

        return input, target, target_scale2, target_scale3

    def __len__(self):
        return len(self.image_filenames)
