import os
import multiprocessing as mp
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('thread_num', type=int)
parser.add_argument('--video_dir', type=str, default='./data/anet/anet_train')
parser.add_argument('--output_dir', type=str, default='./data/anet/afsd_anet_768frames/training')
parser.add_argument('--max_frame', type=int, default=768)
args = parser.parse_args()

thread_num = args.thread_num
video_dir = args.video_dir
output_dir = args.output_dir
        if os.path.exists(target_file):
            print('{} exists, skip.'.format(target_file))
            continue
        cap = cv2.VideoCapture(os.path.join(video_dir, file))
        max_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ratio = max_frame * 1.0 / frame_num
        target_fps = max_fps * ratio

        if file_name[0]=='0':
            file_name=file_name[7:]
        elif file_name[0]=='v':
            file_name=file_name[2:]
        cmd_0='mkdir -p {}'.format(
            os.path.join(output_dir,file_name)
        )

        target_file=os.path.join(output_dir, file_name )
        cmd = 'ffmpeg -i {} -q:v 1 -r {} {}/image_%5d.jpg'.format(
            os.path.join(video_dir, file),
            target_fps,
            target_file
        )
        print("cmd:",cmd)
        os.system(cmd_0)
        os.system(cmd)


processes = []
video_num = len(files)
per_process_video_num = video_num // thread_num

for i in range(thread_num):
    if i == thread_num - 1:
        sub_files = files[i * per_process_video_num:]
    else:
        sub_files = files[i * per_process_video_num: (i + 1) * per_process_video_num]
    p = mp.Process(target=sub_processor, args=(i, sub_files))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
