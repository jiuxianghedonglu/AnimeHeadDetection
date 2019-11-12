# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-11-08 10:46:25 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-11-08 10:46:25 
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import box_iou


class F1Evalutor(object):
    def __init__(self, score_thresh=0.9, iou_thresh=0.7):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.reset()

    def reset(self):
        self.tp = 0
        self.tp_fp = 0
        self.tp_fn = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.eps = 1e-9

    def update(self, pred_boxes_list, pred_scores_list, gt_boxes_list):
        for _, (pred_boxes, pred_scores, gt_boxes) in enumerate(zip(pred_boxes_list, pred_scores_list, gt_boxes_list)):
            pred_boxes = pred_boxes[pred_scores >= self.score_thresh]
            if pred_boxes.size(0) == 0 or gt_boxes.size(0) == 0:
                tp = 0
            else:
                iou = box_iou(gt_boxes, pred_boxes)
                max_iou, _ = iou.max(dim=1)
                tp = (max_iou > self.iou_thresh).sum().item()
            self.tp += tp
            self.tp_fp += pred_boxes.size(0)
            self.tp_fn += gt_boxes.size(0)
            self.precision = self.tp/(self.tp_fp+self.eps)
            self.recall = self.tp/(self.tp_fn+self.eps)
            self.f1_score = 2*self.precision * \
                self.recall/(self.precision+self.recall+self.eps)

    def get_log_str(self):
        logs = 'Precision:{:.5f}\tRecall:{:.5f}\tF1-Score:{:.5f}'.format(
            self.precision, self.recall, self.f1_score)
        return logs


if __name__ == '__main__':
    pass
