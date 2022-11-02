import argparse
from os import listdir
import torch
import cv2 as cv
from math import exp
import torch.nn.functional as F
# import pytorch_ssim
from torchvision import transforms
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import codecs

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


input_root_path = 'output/hm_cbr_high/'
hevc_root_path = '/home/zhr/Project/VECNN_MF/datasets/test/hm_cbr_high_frames/'
gt_root_path = '/home/zhr/Project/CBREN_LD/datasets/test/hevc_sequence_gt_frames/'
folder_list = sorted(listdir(input_root_path))

for sequence in folder_list:  # A B C D E
    if sequence == 'G':
        pass
    else:
        sub_folder_list = sorted(listdir('{}/{}'.format(input_root_path, sequence)))
        for video in sub_folder_list:  # PeopleOnStreet_2560x1600_30_crop Traffic_2560x1600_30_crop
            input_path = '{}{}/{}/'.format(input_root_path, sequence, video)
            hevc_path = '{}{}/{}/'.format(hevc_root_path, sequence, video)
            gt_path = '{}{}/{}/'.format(gt_root_path, sequence, video)
            img_list = []
            for frame in listdir(input_path):
                img_list.append(listdir(input_path))
            print(input_path, ' ', len(img_list), 'test images')
            input_ssim_avg = 0
            hevc_ssim_avg = 0

            input_psnr_avg = 0
            hevc_psnr_avg = 0
            index = 0
            for i in listdir(input_path):
                if is_image_file(i):
                    input_name = '{}{}'.format(input_path, i)
                    hevc_name = '{}{}'.format(hevc_path, i)
                    gt_name = '{}{}'.format(gt_path, i)

                    input_img = cv.imread(input_name)
                    hevc_img = cv.imread(hevc_name)
                    gt_img = cv.imread(gt_name)

                    # input_img = cv.cvtColor(input_img, cv.COLOR_BGR2YUV)
                    # input_img = input_img[:, :, 0]
                    # gt_img = cv.cvtColor(gt_img, cv.COLOR_BGR2YUV)
                    # gt_img = gt_img[:, :, 0]
                    # hevc_img = cv.cvtColor(hevc_img, cv.COLOR_BGR2YUV)
                    # hevc_img = hevc_img[:, :, 0]

                    input_psnr = psnr(transforms.ToTensor()(input_img),
                                      transforms.ToTensor()(gt_img))
                    hevc_psnr = psnr(transforms.ToTensor()(hevc_img),
                                     transforms.ToTensor()(gt_img))

                    # input_img = Image.open(input_name)
                    # hevc_img = Image.open(hevc_name)
                    # gt_img = Image.open(gt_name)
                    #
                    # input_img_ybr = input_img.convert('YCbCr')
                    # input_img, _, _ = input_img_ybr.split()
                    # hevc_img_ybr = hevc_img.convert('YCbCr')
                    # hevc_img, _, _ = hevc_img_ybr.split()
                    # gt_img_ybr = gt_img.convert('YCbCr')
                    # gt_img, _, _ = gt_img_ybr.split()
                    #
                    # input_psnr = psnr(transforms.ToTensor()(input_img),
                    #                   transforms.ToTensor()(gt_img))
                    # hevc_psnr = psnr(transforms.ToTensor()(hevc_img),
                    #                  transforms.ToTensor()(gt_img))

                    input_psnr_avg += input_psnr
                    hevc_psnr_avg += hevc_psnr
                    print(input_name)
                    print("%.4f" % input_psnr)

                    input_ssim = ssim(input_img,
                                      gt_img, multichannel=True)
                    hevc_ssim = ssim(hevc_img,
                                     gt_img, multichannel=True)

                    input_ssim_avg += input_ssim
                    hevc_ssim_avg += hevc_ssim
                    # print(input_name)
                    print("%.4f" % input_ssim)

            input_psnr_avg = input_psnr_avg / len(listdir(input_path))
            hevc_psnr_avg = hevc_psnr_avg / len(listdir(input_path))
            psnr_increase = input_psnr_avg - hevc_psnr_avg
            print('input_psnr_avg:', "%.4f" % input_psnr_avg)
            print('hevc_psnr_avg:', "%.4f" % hevc_psnr_avg)
            print('psnr_increase:', "%.4f" % psnr_increase)

            input_ssim_avg = input_ssim_avg / len(listdir(input_path))
            hevc_ssim_avg = hevc_ssim_avg / len(listdir(input_path))
            ssim_increase = input_ssim_avg - hevc_ssim_avg
            print('input_ssim_avg:', "%.4f" % input_ssim_avg)
            print('hevc_ssim_avg:', "%.4f" % hevc_ssim_avg)
            print('ssim_increase:', "%.4f" % ssim_increase)
            with open("result_high.txt", "a") as f:
                f.write(input_path)
                f.write('\npsnr_increase:{}'.format(psnr_increase))
                f.write('\nssim_increase:{}\n'.format(ssim_increase))
