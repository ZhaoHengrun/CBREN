from os import *
import cv2 as cv
from math import exp
import torch.nn.functional as F
import pytorch_ssim
from torchvision import transforms
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim

input_path = '../results/cqp_baseline/D/BasketballPass_416x240_50/output/'
hevc_path = '../results/cqp_baseline/D/BasketballPass_416x240_50/input/'
gt_path = '../results/cqp_baseline/D/BasketballPass_416x240_50/gt/'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


input_ssim_avg = 0
hevc_ssim_avg = 0

for i in listdir(input_path):
    if is_image_file(i):
        input_name = '{}{}'.format(input_path, i)
        hevc_name = '{}{}'.format(hevc_path, i)
        gt_name = '{}{}'.format(gt_path, i)

        input_img = cv.imread(input_name)
        hevc_img = cv.imread(hevc_name)
        gt_img = cv.imread(gt_name)

        # input_ssim = pytorch_ssim.ssim(transforms.ToTensor()(input_img).unsqueeze(0),
        #                                transforms.ToTensor()(gt_img).unsqueeze(0))
        # hevc_ssim = pytorch_ssim.ssim(transforms.ToTensor()(hevc_img).unsqueeze(0),
        #                               transforms.ToTensor()(gt_img).unsqueeze(0))
        # input_ssim = compare_ssim(input_img,
        #                           gt_img, multichannel=True)
        # hevc_ssim = compare_ssim(hevc_img,
        #                          gt_img, multichannel=True)

        input_img = cv.cvtColor(input_img, cv.COLOR_BGR2YUV)
        input_img = input_img[:, :, 0]
        gt_img = cv.cvtColor(gt_img, cv.COLOR_BGR2YUV)
        gt_img = gt_img[:, :, 0]
        hevc_img = cv.cvtColor(hevc_img, cv.COLOR_BGR2YUV)
        hevc_img = hevc_img[:, :, 0]

        input_ssim = ssim(input_img,
                          gt_img, multichannel=True)
        hevc_ssim = ssim(hevc_img,
                         gt_img, multichannel=True)

        input_ssim_avg += input_ssim
        hevc_ssim_avg += hevc_ssim
        print(input_name)
        print("%.4f" % input_ssim)

input_ssim_avg = input_ssim_avg / len(listdir(input_path))
hevc_ssim_avg = hevc_ssim_avg / len(listdir(input_path))
print('input_ssim_avg:', "%.4f" % input_ssim_avg)
print('hevc_ssim_avg:', "%.4f" % hevc_ssim_avg)
print('ssim_increase:', "%.4f" % (input_ssim_avg - hevc_ssim_avg))
