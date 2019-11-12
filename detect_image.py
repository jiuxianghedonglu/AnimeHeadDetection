# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-11-12 11:04:20 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-11-12 11:04:20 
"""
import os
import sys

import torch

from predictor import Predictor


if __name__ == '__main__':
    try:
        img_path = sys.argv[1]
        img_name = os.path.basename(img_path)
    except:
        print('Please enter a file path')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = Predictor(device=device)

    img = predictor.read_img(img_path)
    x = predictor.process_img(img)
    predictions = predictor.predict(x)
    img = predictor.display_boxes(img, predictions)
    img.save('demo/{}'.format(img_name))
    img.show()
