# -*- coding: utf-8 -*-
"""
  @Author: zzn
  @Date: 2019-11-05 12:21:26
  @Last Modified by:   zzn
  @Last Modified time: 2019-11-05 12:21:26
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ANIMEDataset, collate_fn
from evalutor import F1Evalutor
from model import fasterrcnn_resnet_fpn
from transforms import get_transforms
from utils import AverageMeter


def parse_args():
    def str2bool(v):
        if v.lower() in ['yes', 'true']:
            return True
        else:
            return False
    parser = argparse.ArgumentParser(
        description='Train fasterrcnn_resnet_fpn network.')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr_steps', nargs='+', type=int,
                        default=[10, 15, 20, 25, 30])
    parser.add_argument('--lr_gamma', default=0.5, type=float)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default='0.9')
    parser.add_argument('--train_labeled_file', type=str,
                        default='data/tr_label.csv')
    parser.add_argument('--val_labeled_file', type=str,
                        default='data/val_label.csv')
    parser.add_argument('--img_dir', type=str, default='data/imgs/')
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--long_size', type=int, default=1024)
    parser.add_argument('--works', type=int, default=4)
    parser.add_argument('--logs', type=str, default='checkpoints/logs.txt')
    parser.add_argument('--resume', type=str2bool, default='False')
    args = parser.parse_args()
    print(args)
    return args


def train_one_epoch(model, dataloader, optimizer, epoch, log_file):
    model.train()
    loss_sum_meter = AverageMeter()
    loss_classifier_meter = AverageMeter()
    loss_box_reg_meter = AverageMeter()
    loss_objectness_meter = AverageMeter()
    loss_rpn_box_reg_meter = AverageMeter()
    for i, (x, targets) in enumerate(dataloader):
        x = list(t.to(args.device) for t in x)
        labels = [{k: v.to(args.device) for k, v in t.items()}
                  for t in targets]
        loss_dict = model(x, labels)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        loss_sum = sum([v for _, v in loss_dict.items()])
        loss_sum_meter.update(loss_sum, len(x))
        loss_classifier_meter.update(loss_dict['loss_classifier'], len(x))
        loss_box_reg_meter.update(loss_dict['loss_box_reg'], len(x))
        loss_objectness_meter.update(loss_dict['loss_objectness'], len(x))
        loss_rpn_box_reg_meter.update(
            loss_dict['loss_rpn_box_reg'], len(x))
        if (i+1) % args.print_freq == 0:
            log = ('[Train]Epoch: [{0}][{1}/{2}]\t'
                   'Loss_sum: {loss_sum.val: .6f}({loss_sum.avg: .6f})\t'
                   'Cls: {loss_classifier.val: .6f}({loss_classifier.avg: .6f})\t'
                   'Box: {loss_box_reg.val: .6f}({loss_box_reg.avg: .6f})\t'
                   'Obj: {loss_objectness.val: .6f}({loss_objectness.avg: .6f})\t'
                   'RPN {loss_rpn_box_reg.val: .6f}({loss_rpn_box_reg.avg: .6f})'.format(
                       epoch, i+1, len(dataloader), loss_sum=loss_sum_meter, loss_classifier=loss_classifier_meter,
                       loss_box_reg=loss_box_reg_meter, loss_objectness=loss_objectness_meter, loss_rpn_box_reg=loss_rpn_box_reg_meter))
            log_file.write(log[0]+'\n')
            print(log)


def validation(model, dataloader, epoch, log_file):
    model.eval()
    evalutor = F1Evalutor()
    with torch.no_grad():
        for _, (x, targets) in enumerate(dataloader):
            x = list(t.to(args.device) for t in x)
            gt_boxes_list = [t['boxes'].to(args.device) for t in targets]
            pred_boxes_list = []
            pred_scores_list = []
            predictions = model(x)
            for p in predictions:
                pred_boxes_list.append(p['boxes'])
                pred_scores_list.append(p['scores'])
                evalutor.update(pred_boxes_list,
                                pred_scores_list, gt_boxes_list)
        log = '[Validation]Epoch: [{}]\t{}'.format(
            epoch, evalutor.get_log_str())
        print(log)

    return evalutor.f1_score


if __name__ == '__main__':
    args = parse_args()
    log_file = open(args.logs, 'w', encoding='utf-8')
    train_dataset = ANIMEDataset(
        labeled_file=args.train_labeled_file, img_dir=args.img_dir, transforms=get_transforms(True), long_size=args.long_size)
    val_dataset = ANIMEDataset(
        labeled_file=args.val_labeled_file, img_dir=args.img_dir, transforms=get_transforms(False), long_size=args.long_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.works, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.works, collate_fn=collate_fn)
    model = fasterrcnn_resnet_fpn(
        resnet_name=args.backbone)
    model = model.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    best_f1 = -100
    start_epoch = 0
    if args.resume:
        checkpoints = torch.load('checkpoints/last_checkpoint.pt')
        start_epoch = checkpoints['epoch']
        model.load_state_dict(checkpoints['weights'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        lr_scheduler.load_state_dict(checkpoints['lr_scheduler'])
    for e in range(start_epoch+1, args.epochs+1):
        train_one_epoch(model, train_loader, optimizer, e, log_file)
        lr_scheduler.step()
        val_f1 = validation(model, val_loader, e, log_file)
        states = {
            'epoch': e,
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }
        torch.save(states,
                   'checkpoints/last_checkpoint.pt')
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(),
                       'checkpoints/weights/best_model.pt')
        if e % args.save_freq == 0:
            torch.save(model.state_dict(),
                       'checkpoints/weights/epoch_{}_f1_{:.5f}.pt'.format(e, val_f1))
    log_file.close()
