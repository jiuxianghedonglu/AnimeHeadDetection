# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-11-05 15:26:18 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-11-05 15:26:18 
"""


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
