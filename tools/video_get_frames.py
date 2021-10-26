from os import *
from os.path import join
import cv2

video_path = '../datasets/gt_mp4/D/'
save_path = '../datasets/gt_frames/D/'

current_save_path = ''

for i in listdir(video_path):
    print(i)
    full_video_name = i.split('.')
    video_name = full_video_name[0]

    cap = cv2.VideoCapture('{}{}.mp4'.format(video_path, video_name))
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = 2000

    # for j in range(0, total_frames):
    for j in range(1, total_frames + 1):
        if j % 1 == 0:
            ret, frame = cap.read()
            if not ret:
                break
            current_save_path = '{}{}/'.format(save_path, video_name)
            if not path.exists(current_save_path):
                mkdir(current_save_path)
            print('writting video[{}]frame[{}]'.format(video_name, j))
            # cv2.imwrite('{}{}.png'.format(current_save_path, j), frame)
            cv2.imwrite('{}{}.png'.format(current_save_path, "%03d" % j), frame)

        # cv2.imwrite('{}{}.png'.format(current_save_path, j), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

print('finish')
