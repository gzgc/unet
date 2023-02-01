from __future__ import print_function, division
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
try:
    from itertools import  ifilterfalse
except ImportError:
    from itertools import  filterfalse as ifilterfalse

    
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum() # 假设 gts = 5   gt_sorted=[1,0,1,1,0,0,1,1]
    intersection = gts - gt_sorted.float().cumsum(0) #gt_sorted.float().cumsum(0) = [1,1,2,3,3,3,4,4] #构建单调非减序列  则gts- = [4,4,3,2,2,2,1,1]
    union = gts + (1 - gt_sorted).float().cumsum(0)# 1 - gt_sorted = [0,1,0,0,1,1,0,0]  (1 - gt_sorted).float().cumsum(0)=[0,1,1,1,2,3,3,3]  union = [5,6,6,6,7,8,8,8]
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
def lovasz_softmax_flat(probas, labels, classes='present'): 
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    
    # 类别数
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes 
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c 【1，1，1，0，0，0，0，1，1】
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]#预测值预测为第c类的概率 【0.1，0.8，0.7，0.1  .....】
        errors = (Variable(fg) - class_pred).abs()  # 对每个点预测错误值
        errors_sorted, perm = torch.sort(errors, 0, descending=True) # 对误差排序 ，descending=True 从大到小排序，返回值误差从大到小，perm表示位置信息
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
def flatten_probas(probas, labels, ignore=None):
    # 把输出 和标签 拉成【【】，【】，【】，【】】 和 【0，1，0，2，0，1】
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class LovaszLossSoftmax(nn.Module):
    def __init__(self):
        super(LovaszLossSoftmax, self).__init__()

    def forward(self, input, target):
        out = F.softmax(input, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
        return loss
