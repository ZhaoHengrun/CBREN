from os import *
import torch
import cv2 as cv
from math import exp
import torch.nn.functional as F
import pytorch_ssim
from torchvision import transforms
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim
from PIL import Image

input_path = '../datasets/test/LIVE1_qf10/'
compressed_path = '../datasets/test/LIVE1_qf10/'
gt_path = '../datasets/test/LIVE1_gt/'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


input_ssim_avg = 0
compressed_ssim_avg = 0

input_psnr_avg = 0
compressed_psnr_avg = 0

for i in listdir(input_path):
    if is_image_file(i):
        input_name = '{}{}'.format(input_path, i)
        compressed_name = '{}{}'.format(compressed_path, i)
        gt_name = '{}{}'.format(gt_path, i)

        input_img = cv.imread(input_name)
        compressed_img = cv.imread(compressed_name)
        gt_img = cv.imread(gt_name)

        input_img = cv.cvtColor(input_img, cv.COLOR_BGR2YUV)
        input_img = input_img[:, :, 0]
        gt_img = cv.cvtColor(gt_img, cv.COLOR_BGR2YUV)
        gt_img = gt_img[:, :, 0]
        compressed_img = cv.cvtColor(compressed_img, cv.COLOR_BGR2YUV)
        compressed_img = compressed_img[:, :, 0]

        input_psnr = psnr(transforms.ToTensor()(input_img),
                          transforms.ToTensor()(gt_img))
        compressed_psnr = psnr(transforms.ToTensor()(compressed_img),
                         transforms.ToTensor()(gt_img))

        # input_img = Image.open(input_name)
        # compressed_img = Image.open(compressed_name)
        # gt_img = Image.open(gt_name)
        #
        # input_img_ybr = input_img.convert('YCbCr')
        # input_img, _, _ = input_img_ybr.split()
        # compressed_img_ybr = compressed_img.convert('YCbCr')
        # compressed_img, _, _ = compressed_img_ybr.split()
        # gt_img_ybr = gt_img.convert('YCbCr')
        # gt_img, _, _ = gt_img_ybr.split()
        #
        # input_psnr = psnr(transforms.ToTensor()(input_img),
        #                   transforms.ToTensor()(gt_img))
        # compressed_psnr = psnr(transforms.ToTensor()(compressed_img),
        #                  transforms.ToTensor()(gt_img))

        input_psnr_avg += input_psnr
        compressed_psnr_avg += compressed_psnr
        print(input_name)
        print("%.4f" % input_psnr)

        input_ssim = ssim(input_img,
                          gt_img, multichannel=True)
        compressed_ssim = ssim(compressed_img,
                         gt_img, multichannel=True)

        input_ssim_avg += input_ssim
        compressed_ssim_avg += compressed_ssim
        print(input_name)
        print("%.4f" % input_ssim)

input_psnr_avg = input_psnr_avg / len(listdir(input_path))
compressed_psnr_avg = compressed_psnr_avg / len(listdir(input_path))
print('input_psnr_avg:', "%.4f" % input_psnr_avg)
print('compressed_psnr_avg:', "%.4f" % compressed_psnr_avg)
print('psnr_increase:', "%.4f" % (input_psnr_avg - compressed_psnr_avg))

input_ssim_avg = input_ssim_avg / len(listdir(input_path))
compressed_ssim_avg = compressed_ssim_avg / len(listdir(input_path))
print('input_ssim_avg:', "%.4f" % input_ssim_avg)
print('compressed_ssim_avg:', "%.4f" % compressed_ssim_avg)
print('ssim_increase:', "%.4f" % (input_ssim_avg - compressed_ssim_avg))
