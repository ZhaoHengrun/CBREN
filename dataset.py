import os
from os import listdir
import random
from random import randint

import cv2 as cv
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import time


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class DataExpasion(object):
    def __call__(self, img_input):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        if hflip:
            img_input = img_input[:, ::-1, :]
        if vflip:
            img_input = img_input[::-1, :, :]
        if rot90:
            img_input = img_input.transpose(1, 0, 2)
        return img_input


class TwinDataExpasion(object):
    def __init__(self, multi_frame):
        self.multi_frame = multi_frame

    def __call__(self, img_input):
        img_input_1 = img_input['target']
        img_input_2 = img_input['input']
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5
        if self.multi_frame is True:
            num, r, c, ch = img_input_2.shape
            if hflip:
                img_input_1 = img_input_1[:, ::-1, :]
                for idx in range(num):
                    img_input_2[idx, :, :, :] = img_input_2[idx, :, ::-1, :]
            if vflip:
                img_input_1 = img_input_1[::-1, :, :]
                for idx in range(num):
                    img_input_2[idx, :, :, :] = img_input_2[idx, ::-1, :, :]
            if rot90:
                img_input_1 = img_input_1.transpose(1, 0, 2)
                img_input_2 = img_input_2.transpose(0, 2, 1, 3)
        else:
            if hflip:
                img_input_1 = img_input_1[:, ::-1, :]
                img_input_2 = img_input_2[:, ::-1, :]
            if vflip:
                img_input_1 = img_input_1[::-1, :, :]
                img_input_2 = img_input_2[::-1, :, :]
            if rot90:
                img_input_1 = img_input_1.transpose(1, 0, 2)
                img_input_2 = img_input_2.transpose(1, 0, 2)

        return {'target': img_input_1, 'input': img_input_2}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img_input):
        h, w = img_input.shape[0], img_input.shape[1]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        new_input = img_input[top:top + new_h, left: left + new_w, :]
        return new_input


class TwinRandomCrop(object):
    def __init__(self, output_size, multi_frame=False):
        self.multi_frame = multi_frame
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img_input):
        img_input_1 = img_input['target']
        img_input_2 = img_input['input']
        h = img_input_1.shape[0]
        w = img_input_1.shape[1]
        new_h = self.output_size[0]
        new_w = self.output_size[1]

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        new_input_1 = img_input_1[top:top + new_h, left: left + new_w, :]
        if self.multi_frame is False:
            new_input_2 = img_input_2[top:top + new_h, left: left + new_w, :]
        else:
            new_input_2 = img_input_2[:, top:top + new_h, left: left + new_w, :]
        return {'target': new_input_1, 'input': new_input_2}


class NumpyToTensor(object):
    def __init__(self, multi_frame=False):
        self.multi_frame = multi_frame

    def __call__(self, numpy_input):
        numpy_input = numpy_input / 255.0
        numpy_input = torch.from_numpy(numpy_input).float()
        if self.multi_frame is True:
            return numpy_input.permute(0, 3, 1, 2)
        else:
            return numpy_input.permute(2, 0, 1)


class TrainDatasetMultiFrame(data.Dataset):
    def __init__(self, target_path, input_path):
        self.target_path = target_path
        self.input_path = input_path
        self.folder_list = sorted(listdir(self.target_path))
        self.img_list = []
        for _f in self.folder_list:
            target_folder_path = '{}{}/'.format(self.target_path, _f)
            self.img_list.append(sorted(listdir(target_folder_path)))

        self.crop_size = 96
        # self.crop_size = 128
        self.twin_transform = transforms.Compose([
            TwinRandomCrop(self.crop_size, multi_frame=True),
            TwinDataExpasion(multi_frame=True)
        ])

    def __len__(self):
        # return len(self.folder_list)
        # return 1000
        return 60000

    def __getitem__(self, idx):
        folder_index = randint(0, (len(self.folder_list) - 1))
        img_index = randint(2, len(self.img_list[folder_index]) - 3)
        target_img_name = '{}{}/{}.png'.format(self.target_path, self.folder_list[folder_index],
                                               img_index)

        target_img = cv.imread(target_img_name)
        h, w, ch = target_img.shape

        input_img = np.zeros((5, h, w, ch))
        input_img_name_1 = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index - 2)
        input_img[0, :, :, :] = cv.imread(input_img_name_1)
        input_img_name_2 = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index - 1)
        input_img[1, :, :, :] = cv.imread(input_img_name_2)
        input_img_name_3 = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index)
        input_img[2, :, :, :] = cv.imread(input_img_name_3)
        input_img_name_4 = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index + 1)
        input_img[3, :, :, :] = cv.imread(input_img_name_4)
        input_img_name_5 = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index + 2)
        input_img[4, :, :, :] = cv.imread(input_img_name_5)
        sample = {'target': target_img, 'input': input_img}
        sample = self.twin_transform(sample)
        target_img = sample['target']
        input_img = sample['input']

        target_tensor = NumpyToTensor()(target_img.copy())  # [3, 128, 128]
        input_tensor = NumpyToTensor(multi_frame=True)(input_img.copy())  # [5, 3, 128, 128]
        return input_tensor, target_tensor


class ValidDatasetMultiFrame(data.Dataset):
    def __init__(self, target_path, input_path):
        self.target_path = target_path
        self.input_path = input_path
        self.folder_list = sorted(listdir(self.target_path))
        self.img_list = []
        for _f in self.folder_list:
            target_folder_path = '{}{}/'.format(self.target_path, _f)
            self.img_list.append(sorted(listdir(target_folder_path)))

    def __len__(self):
        return 200

    def __getitem__(self, idx):
        folder_index = idx // 20
        img_index = idx
        if idx <= 3:
            img_index = 2
        if idx >= (len(self.img_list) - 3):
            img_index = len(self.img_list) - 3
        target_img_name = '{}{}/{}.png'.format(self.target_path, self.folder_list[folder_index],
                                               img_index % 20)

        target_img = cv.imread(target_img_name)
        h, w, ch = target_img.shape

        input_img = np.zeros((5, h, w, ch))
        input_img_name_1 = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index - 2)
        input_img[0, :, :, :] = cv.imread(input_img_name_1)
        input_img_name_2 = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index - 1)
        input_img[1, :, :, :] = cv.imread(input_img_name_2)
        input_img_name_3 = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index)
        input_img[2, :, :, :] = cv.imread(input_img_name_3)
        input_img_name_4 = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index + 1)
        input_img[3, :, :, :] = cv.imread(input_img_name_4)
        input_img_name_5 = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index + 2)
        input_img[4, :, :, :] = cv.imread(input_img_name_5)

        target_tensor = NumpyToTensor()(target_img.copy())  # [3, 128, 128]
        input_tensor = NumpyToTensor(multi_frame=True)(input_img.copy())  # [5, 3, 128, 128]
        return input_tensor, target_tensor


class TrainDatasetSingleFrame(data.Dataset):
    def __init__(self, target_path, input_path):
        self.target_path = target_path
        self.input_path = input_path
        self.folder_list = sorted(listdir(self.target_path))
        self.img_list = []

        for _f in self.folder_list:
            target_folder_path = '{}{}/'.format(self.target_path, _f)
            self.img_list.append(sorted(listdir(target_folder_path)))

        self.crop_size = 96
        self.twin_transform = transforms.Compose([
            TwinRandomCrop(self.crop_size, multi_frame=False),
            TwinDataExpasion(multi_frame=False)
        ])

    def __len__(self):
        # return len(self.lis)
        return 60000
        # return 10

    def __getitem__(self, index):
        folder_index = randint(0, (len(self.folder_list) - 1))
        img_index = randint(2, len(self.img_list[folder_index]) - 3)
        target_img_name = '{}{}/{}'.format(self.target_path, self.folder_list[folder_index],
                                           self.img_list[folder_index][img_index])
        input_img_name = '{}{}/{}'.format(self.input_path, self.folder_list[folder_index],
                                          self.img_list[folder_index][img_index])

        target_img = cv.imread(target_img_name)
        input_img = cv.imread(input_img_name)

        sample = {'target': target_img, 'input': input_img}
        sample = self.twin_transform(sample)
        target_img = sample['target']
        input_img = sample['input']

        target_tensor = NumpyToTensor()(target_img.copy())
        input_tensor = NumpyToTensor()(input_img.copy())
        return input_tensor, target_tensor


class ValidDatasetSingleFrame(data.Dataset):
    def __init__(self, target_path, input_path):
        self.target_path = target_path
        self.input_path = input_path
        self.folder_list = sorted(listdir(self.target_path))
        self.img_list = []
        for _f in self.folder_list:
            target_folder_path = '{}{}/'.format(self.target_path, _f)
            self.img_list.append(sorted(listdir(target_folder_path)))

    def __len__(self):
        return 200

    def __getitem__(self, idx):
        folder_index = idx // 20
        img_index = idx
        if idx <= 3:
            img_index = 2
        if idx >= (len(self.img_list) - 3):
            img_index = len(self.img_list) - 3
        target_img_name = '{}{}/{}.png'.format(self.target_path, self.folder_list[folder_index],
                                               img_index % 20)

        target_img = cv.imread(target_img_name)
        # h, w, ch = target_img.shape

        input_img_name = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index)
        input_img = cv.imread(input_img_name)


        target_tensor = NumpyToTensor()(target_img.copy())  # [3, 128, 128]
        input_tensor = NumpyToTensor()(input_img.copy())  # [3, 128, 128]
        return input_tensor, target_tensor

# class TrainDataset(data.Dataset):
#     def __init__(self, target_path, input_path):
#         super(TrainDataset, self).__init__()
#         self.target_path = target_path
#         self.input_path = input_path
#         self.lis = sorted(os.listdir(target_path))
#         # self.crop_size = 43
#         # self.transform = transforms.Compose([
#         #     RandomCrop(self.crop_size),
#         #     DataExpasion(),
#         # ])
#         self.twin_transform = transforms.Compose([
#             # TwinRandomCrop(self.crop_size),
#             TwinDataExpasion(multi_frame=False),
#         ])
#         self.to_tensor = NumpyToTensor
#
#     def __getitem__(self, index):
#         img_target_path = '{}{}'.format(self.target_path, self.lis[index])
#         img_input_path = '{}{}'.format(self.input_path, self.lis[index])
#         target_img = cv.imread(img_target_path)
#         input_img = cv.imread(img_input_path)
#         sample = [target_img, input_img]
#         sample = self.twin_transform(sample)
#         target_img = sample[0]
#         input_img = sample[1]
#
#         target_tensor = transforms.ToTensor()(target_img.copy())
#         input_tensor = transforms.ToTensor()(input_img.copy())
#         return input_tensor, target_tensor
#
#     def __len__(self):
#         return len(self.lis)
