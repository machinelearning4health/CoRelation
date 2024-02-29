import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


def loss_fn(logits, label, loss_config, avg_label_num=None, with_ori_format=False):
    logits = logits.contiguous()
    label = label.contiguous().type_as(logits)
    with torch.cuda.amp.autocast(enabled=False):
        if not loss_config or loss_config['name'] == 'ce':
            # cross entropy
            if avg_label_num is not None:
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), label.view(-1), reduction='sum')
                loss = loss / avg_label_num / label.shape[0]
            else:
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), label.view(-1))
        elif loss_config['name'] == 'sigce':
            if avg_label_num is not None:
                loss = F.binary_cross_entropy(logits.view(-1), label.view(-1), reduction='sum')
                loss = loss / avg_label_num / label.shape[0]
            elif with_ori_format:
                loss = F.binary_cross_entropy(logits, label, reduction='none')
            else:
                loss = F.binary_cross_entropy(logits.view(-1), label.view(-1))
    return loss
