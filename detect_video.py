# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-11-12 11:04:36 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-11-12 11:04:36 
"""
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from predictor import Predictor

if __name__ == '__main__':
    try:
        video_path = sys.argv[1]
        video_name = os.path.basename(video_path)
    except:
        print('Please enter a file path')

    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print('Unable to read')
    fps = 20
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('demo/detect_video_{}.avi'.format(video_name), cv2.VideoWriter_fourcc(
        'M', 'P', '4', '2'), fps, (frame_width, frame_height))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = Predictor(device=device)

    while(True):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            x = predictor.process_img(img)
            predictions = predictor.predict(x)
            img = predictor.display_boxes(img, predictions)
            save_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            out.write(save_frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
