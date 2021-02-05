import os
from os.path import dirname
import cv2

def sort_num(fname):
    num = fname.split(".png")[0]
    num = num.split("-")[-1]
    
    return int(num)

dir_name = "./data/"
frames = [os.path.join(dir_name,i) for i in  os.listdir(dir_name) if "frame" in i]
frames = sorted(frames, key=sort_num)

frames = [cv2.imread(i) for i in frames]


if frames:
    frame_size = tuple(int(i) for i in frames[0].shape[:2])
    print(frame_size)
    # frame_size = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(dir_name,'output.mp4'),fourcc, 20.0, frame_size, True)

    for frame in frames:
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    out.release()