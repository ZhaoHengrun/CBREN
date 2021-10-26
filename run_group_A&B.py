from __future__ import print_function
import argparse
from os import listdir
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image
from torchvision import transforms
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
# from dataset import *
from math import exp
import torch.nn.functional as F
# import pytorch_ssim
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--submit_mode', default=False, help='Test all sequences directly')
parser.add_argument('--input_path', type=str,
                    default='datasets/hevc_cbr_low_frames/D/BasketballPass_416x240_50/',
                    help='input path to use')
parser.add_argument('--output_path', default='results/hevc_cbr_low/',
                    type=str,
                    help='where to save the output image')
parser.add_argument('--gt_path', type=str,
                    default='datasets/gt_frames/D/BasketballPass_416x240_50/',
                    help='input path to use')
parser.add_argument('--model', type=str, default='checkpoints/cbr_low.pth',
                    help='model file to use')
parser.add_argument('--calc_on_y', default=False, action='store_true', help='calc on y channel')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


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


def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor[:, [2, 1, 0], :, :]
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    # input_tensor = cv.cvtColor(input_tensor, cv.COLOR_RGB2BGR)
    cv.imwrite(filename, input_tensor)


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor[:, [2, 1, 0], :, :]
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)


def padding8(img):
    h, w = img.shape[0:2]
    pad_h = 8 - h % 8 if h % 8 != 0 else 0
    pad_w = 8 - w % 8 if w % 8 != 0 else 0
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'edge')
    return img


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def load_img_helf(img_name, pos):
    img = cv.imread(img_name)
    height, weight = img.shape[0], img.shape[1]
    if pos == 0:
        img = img[0:height // 2 + 100, 0:weight // 2 + 100]
    if pos == 1:
        img = img[0:height // 2 + 100, weight // 2 - 100:weight]
    if pos == 2:
        img = img[height // 2 - 100:height, 0:weight // 2 + 100]
    if pos == 3:
        img = img[height // 2 - 100:height, weight // 2 - 100:weight]
    return img


if opt.cuda:
    model = torch.load(opt.model, map_location='cuda:0')
else:
    model = torch.load(opt.model, map_location='cpu')

input_path = opt.input_path
img_list = []
for _f in listdir(input_path):
    img_list.append(listdir(input_path))
print(len(img_list), 'test images')
input_psnr_avg = 0
output_psnr_avg = 0
input_ssim_avg = 0
output_ssim_avg = 0
index = 0
for i in listdir(input_path):
    if is_image_file(i):
        with torch.no_grad():
            index = index + 1
            example_img = cv.imread('{}{}.png'.format(input_path, "%03d" % 1))
            h, w, ch = example_img.shape
            for j in range(4):
                input_img = np.zeros((5, h // 2 + 100, w // 2 + 100, ch))
                input_img_name_1 = '{}{}.png'.format(input_path, "%03d" % (index - 2))
                input_img_name_2 = '{}{}.png'.format(input_path, "%03d" % (index - 1))
                input_img_name_3 = '{}{}.png'.format(input_path, "%03d" % index)
                input_img_name_4 = '{}{}.png'.format(input_path, "%03d" % (index + 1))
                input_img_name_5 = '{}{}.png'.format(input_path, "%03d" % (index + 2))
                if index == 1:
                    input_img_name_1 = '{}{}.png'.format(input_path, "%03d" % index)
                    input_img_name_2 = '{}{}.png'.format(input_path, "%03d" % index)
                    input_img_name_3 = '{}{}.png'.format(input_path, "%03d" % index)
                    input_img_name_4 = '{}{}.png'.format(input_path, "%03d" % (index + 1))
                    input_img_name_5 = '{}{}.png'.format(input_path, "%03d" % (index + 2))
                if index == 2:
                    input_img_name_1 = '{}{}.png'.format(input_path, "%03d" % (index - 1))
                    input_img_name_2 = '{}{}.png'.format(input_path, "%03d" % (index - 1))
                    input_img_name_3 = '{}{}.png'.format(input_path, "%03d" % index)
                    input_img_name_4 = '{}{}.png'.format(input_path, "%03d" % (index + 1))
                    input_img_name_5 = '{}{}.png'.format(input_path, "%03d" % (index + 2))
                if index == len(img_list) - 1:
                    input_img_name_1 = '{}{}.png'.format(input_path, "%03d" % (index - 2))
                    input_img_name_2 = '{}{}.png'.format(input_path, "%03d" % (index - 1))
                    input_img_name_3 = '{}{}.png'.format(input_path, "%03d" % index)
                    input_img_name_4 = '{}{}.png'.format(input_path, "%03d" % index)
                    input_img_name_5 = '{}{}.png'.format(input_path, "%03d" % (index + 1))
                if index == len(img_list):
                    input_img_name_1 = '{}{}.png'.format(input_path, "%03d" % (index - 2))
                    input_img_name_2 = '{}{}.png'.format(input_path, "%03d" % (index - 1))
                    input_img_name_3 = '{}{}.png'.format(input_path, "%03d" % index)
                    input_img_name_4 = '{}{}.png'.format(input_path, "%03d" % index)
                    input_img_name_5 = '{}{}.png'.format(input_path, "%03d" % index)

                input_img[0, :, :, :] = load_img_helf(input_img_name_1, j)
                input_img[1, :, :, :] = load_img_helf(input_img_name_2, j)
                input_img[2, :, :, :] = load_img_helf(input_img_name_3, j)
                input_img[3, :, :, :] = load_img_helf(input_img_name_4, j)
                input_img[4, :, :, :] = load_img_helf(input_img_name_5, j)
                out_img = NumpyToTensor(multi_frame=True)(input_img.copy())
                input = torch.unsqueeze(out_img, dim=0).float().contiguous()

                model.eval()
                if opt.cuda:
                    model = model.cuda()
                    input = input.cuda()

                out = model(input)
                out = out.cpu()

                output_tensor = out.clone().detach()
                output_tensor = output_tensor.squeeze()

                if j == 0:
                    output_tensor_full = torch.zeros(ch, h, w)
                    output_tensor_full[:, 0:h // 2, 0:w // 2] = output_tensor[:, 0:h // 2, 0:w // 2]
                if j == 1:
                    output_tensor_full[:, 0:h // 2, w // 2:w] = output_tensor[:, 0:h // 2, 100:w // 2 + 100]
                if j == 2:
                    output_tensor_full[:, h // 2:h, 0:w // 2] = output_tensor[:, 100:h // 2 + 100, 0:w // 2]
                if j == 3:
                    output_tensor_full[:, h // 2:h, w // 2:w] = output_tensor[:, 100:h // 2 + 100, 100:w // 2 + 100]

            if not os.path.exists(opt.output_path):
                os.makedirs(opt.output_path)
            if not os.path.exists('{}output/'.format(opt.output_path)):
                os.makedirs('{}output/'.format(opt.output_path))
            if not os.path.exists('{}gt/'.format(opt.output_path)):
                os.makedirs('{}gt/'.format(opt.output_path))
            if not os.path.exists('{}input/'.format(opt.output_path)):
                os.makedirs('{}input/'.format(opt.output_path))
            print('processing:', '{}{}.png'.format(opt.output_path, "%03d" % index))
            img_original = cv.imread('{}{}.png'.format(opt.gt_path, "%03d" % index))
            input_center_img = cv.imread('{}{}.png'.format(opt.input_path, "%03d" % index))

            # if opt.save_img is True:
            output_tensor_full_save = output_tensor_full.unsqueeze(0)
            save_image_tensor(output_tensor_full_save,
                              '{}output/{}.png'.format(opt.output_path, "%03d" % index))
            cv.imwrite('{}gt/{}.png'.format(opt.output_path, "%03d" % index), img_original)
            cv.imwrite('{}input/{}.png'.format(opt.output_path, "%03d" % index), input_center_img)

            out_img = output_tensor_full.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(
                torch.uint8).numpy()

            if opt.calc_on_y is True:
                input_center_img = cv.cvtColor(input_center_img, cv.COLOR_BGR2YUV)
                input_center_img = input_center_img[:, :, 0]
                img_original = cv.cvtColor(img_original, cv.COLOR_BGR2YUV)
                img_original = img_original[:, :, 0]
                out_img = cv.cvtColor(out_img, cv.COLOR_BGR2YUV)
                out_img = out_img[:, :, 0]

            input_psnr = calc_psnr(transforms.ToTensor()(input_center_img),
                                   transforms.ToTensor()(img_original))
            output_psnr = calc_psnr(transforms.ToTensor()(out_img),
                                    transforms.ToTensor()(img_original))
            input_ssim = ssim(input_center_img,
                              img_original, multichannel=True)
            output_ssim = ssim(out_img,
                               img_original, multichannel=True)
            if index > 2 or index < len(img_list) - 1:
                input_psnr_avg += input_psnr
                output_psnr_avg += output_psnr
                input_ssim_avg += input_ssim
                output_ssim_avg += output_ssim
            print(output_psnr)

input_psnr_avg = input_psnr_avg / (index - 4)
output_psnr_avg = output_psnr_avg / (index - 4)
input_ssim_avg = input_ssim_avg / (index - 4)
output_ssim_avg = output_ssim_avg / (index - 4)
print('input_psnr_avg:', input_psnr_avg)
print('output_psnr_avg:', output_psnr_avg)
print('psnr_increase:', output_psnr_avg - input_psnr_avg)
print('input_ssim_avg:', input_ssim_avg)
print('output_ssim_avg:', output_ssim_avg)
print('ssim_increase:', output_ssim_avg - input_ssim_avg)
