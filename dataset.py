# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-11-01 10:26:25 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-11-01 10:26:25 
"""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from transforms import Compose, RandomHorizontalFlip, ToTensor


class ANIMEDataset(Dataset):
    def __init__(self, labeled_file, img_dir, transforms, long_size=1024, train_flag=True):
        self.labeled_file = labeled_file
        self.img_dir = img_dir
        self.transforms = transforms
        self.long_size = long_size
        self.train_flag = train_flag
        self.img_paths = self.get_img_paths()
        if self.train_flag:
            self.boxes, self.labels = self.get_boxes_labels()

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        y = None
        if self.train_flag:
            y = {}
            if w > h:
                re_w = self.long_size
                re_h = int(h*re_w/w)
            else:
                re_h = self.long_size
                re_w = int(w*re_h/h)
            re_size = (re_w, re_h)
            img = img.resize(re_size)
            boxes = self.boxes[idx]
            for box in boxes:
                box[0], box[2] = box[0]*re_size[0] / \
                    w, box[2]*re_size[0]/w
                box[1], box[3] = box[1]*re_size[1] / \
                    h, box[3]*re_size[1]/h
            labels = self.labels[idx]
            boxes = torch.tensor(boxes, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)
            y['boxes'] = boxes
            y['labels'] = labels
        x, y = self.transforms(img, y)
        return x, y

    def __len__(self):
        return len(self.img_paths)

    def get_img_paths(self):
        img_paths = []
        with open(self.labeled_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                img_name = line.strip().split(',')[0]
                img_paths.append(os.path.join(self.img_dir, img_name))
        return img_paths

    def get_boxes_labels(self):
        boxes = []
        labels = []
        with open(self.labeled_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                boxes_labels = line.strip().split(',')[1].split(';')
                cur_boxes = []
                cur_labels = []
                for box_label in boxes_labels:
                    box = [float(t)
                           for t in box_label.split(' ')[:-1]]
                    cur_boxes.append(box)
                    cur_labels.append(int(box_label.split(' ')[-1]))
                boxes.append(cur_boxes)
                labels.append(cur_labels)
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    pass
