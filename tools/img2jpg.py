import os
from os import listdir
from PIL import Image

qf = 20
input_path = '../datasets/train_jpg/gt/'
output_path = '../datasets/train_jpg/qf_{}/'.format(qf)

if not os.path.exists(output_path):
    os.makedirs(output_path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


for i in listdir(input_path):
    if is_image_file(i):
        img_path = '{}{}'.format(input_path, i)
        output_dir = '{}{}'.format(output_path, i)
        img = Image.open(img_path)
        print('saving:[{}]'.format(output_dir))
        img.save(output_dir, format='JPEG', quality=qf)
