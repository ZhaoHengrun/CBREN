import os
from os import listdir
from PIL import Image
import cv2 as cv

qf = 10
gt_path = '../datasets/test/LIVE1_gt/'
gt_y_path = '../datasets/test/LIVE1_y_gt/'
output_y_path = '../datasets/test/LIVE1_y_qf10/'

if not os.path.exists(gt_y_path):
    os.makedirs(gt_y_path)
if not os.path.exists(output_y_path):
    os.makedirs(output_y_path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


for i in listdir(gt_path):
    if is_image_file(i):
        gt_dir = '{}{}'.format(gt_path, i)
        gt_y_dir = '{}{}'.format(gt_y_path, i)
        output_y_dir = '{}{}'.format(output_y_path, i)
        img_gt = Image.open(gt_dir)
        img_y_gt = img_gt.convert('L')
        print('saving:[{}]'.format(gt_dir))
        img_y_gt.save(gt_y_dir, format='JPEG', quality=100)
        img_y_gt.save(output_y_dir, format='JPEG', quality=qf)

# for i in listdir(gt_path):
#     if is_image_file(i):
#         gt_dir = '{}{}'.format(gt_path, i)
#         name = i.split('.')
#         i = name[0] + '.' + 'jpg'
#         gt_y_dir = '{}{}'.format(gt_y_path, i)
#         output_y_dir = '{}{}'.format(output_y_path, i)
#
#         img_gt = cv.imread(gt_dir)
#
#         img_gt = cv.cvtColor(img_gt, cv.COLOR_BGR2YUV)
#         img_y_gt = img_gt[:, :, 0]
#         print('saving:[{}]'.format(gt_y_dir))
#         cv.imwrite(gt_y_dir, img_y_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
#         cv.imwrite(output_y_dir, img_y_gt, [cv.IMWRITE_JPEG_QUALITY, 10])
