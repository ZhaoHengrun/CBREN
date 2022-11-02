import os
from os import *
import cv2 as cv
import numpy as np

input_path = '../datasets/train/ntire_gt/'
output_path = '../datasets/train/ntire_gt_crop_compress/'

if not os.path.exists(output_path):
    os.makedirs(output_path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class CropBlocks(object):
    def __call__(self, img_input, output_size, img_name, save_path):
        h, w = img_input.shape[0], img_input.shape[1]
        h_num = h // output_size
        w_num = w // output_size

        for j in range(h_num):
            top = j * output_size
            for k in range(w_num):
                left = k * output_size
                block = img_input[top:top + output_size, left: left + output_size, :]
                block_path = save_path.replace('.png', '_{}_{}.png'.format(j, k))
                cv.imwrite(block_path, block)
        return h_num, w_num


class ShaveEdge(object):
    def __call__(self, img_input, h_decrease, w_decrease):
        h, w = img_input.shape[0], img_input.shape[1]
        new_h, new_w = h - h_decrease, w - w_decrease

        new_input = img_input[h - new_h:h, 0:new_w]
        return new_input


for i in listdir(input_path):
    sub_path = '{}{}/'.format(input_path, i)
    print(sub_path)
    for j in listdir(sub_path):
        if is_image_file(j):
            img_path = '{}{}'.format(sub_path, j)
            output_sub_path = '{}{}/'.format(output_path, i)
            output_name = '{}{}/{}'.format(output_path, i, j)
            if not os.path.exists(output_sub_path):
                os.makedirs(output_sub_path)
            img = cv.imread(img_path)
            # img = ShaveEdge()(img, 1, 1)
            CropBlocks()(img, 128, j, output_name)
            print('saving:[{}]'.format(output_name))

        # cv.imwrite(output_dir, img, [cv.IMWRITE_PNG_COMPRESSION, 0])
        # cv.imwrite(output_dir, img)
        # cv2.imwrite(output_dir, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # img.save(output_dir, format='JPEG', quality=100)
