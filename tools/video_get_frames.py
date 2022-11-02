from os import *
from os.path import join
import cv2

video_path = 'E:/Project/VideoEnhance/VECNN_MF/datasets/valid/gt/'
save_path = 'E:/Project/VideoEnhance/VECNN_MF/datasets/valid/ntire_gt/'


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in [".mkv"])


current_save_path = ''

for i in listdir(video_path):
    if is_video_file(i):
        print(i)
        full_video_name = i.split('.')
        video_name = full_video_name[0]

        cap = cv2.VideoCapture('{}{}.mkv'.format(video_path, video_name))
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = 2000

        for j in range(total_frames):
            if j < 20:
                ret, frame = cap.read()
                if not ret:
                    break
                current_save_path = '{}{}/'.format(save_path, video_name)
                if not path.exists(current_save_path):
                    mkdir(current_save_path)
                print('writting video[{}]frame[{}]'.format(video_name, j))
                cv2.imwrite('{}{}.png'.format(current_save_path, j), frame)
                # cv2.imwrite('{}{}.png'.format(current_save_path, "%03d" % j), frame)

            # cv2.imwrite('{}{}.png'.format(current_save_path, j), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

print('finish')
