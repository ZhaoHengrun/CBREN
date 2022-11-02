from os import *
import cv2

input_path = '../datasets/valid/qp_51_video/'
output_path = '../datasets/valid/qp_51_img/'


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in [".mp4"])


if not path.exists(output_path):
    makedirs(output_path)

for i in listdir(input_path):
    if is_video_file(i):
        cap = cv2.VideoCapture('{}{}'.format(input_path, i))
        ret, frame = cap.read()
        output_name = i.replace('_batch.mp4', '.png')
        # output_name = i.replace('_batch.mp4', '.jpg')
        output_dir = '{}{}'.format(output_path, output_name)

        print('saving:[{}]'.format(output_dir))
        cv2.imwrite(output_dir, frame)
        # cv2.imwrite(output_dir, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # cv2.imwrite(output_dir, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

print('finish')
