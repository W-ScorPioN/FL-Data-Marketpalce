#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

"""
功能：
    1. 对数据集进行投毒攻击，产生被污染的模型，通过设置poison_fraction控制投毒比例
"""

import numpy as np

# 标签投毒
def poison_labels(dataset, poison_fraction):
    num_poison = int(len(dataset) * poison_fraction)
    poison_indices = np.random.choice(len(dataset), num_poison, replace=False)

    for idx in poison_indices:
        original_label = dataset.labels[idx]
        # 将标签随机更改为除原始标签外的其他类别
        possible_labels = list(range(dataset.num_classes))
        possible_labels.remove(original_label)
        poisoned_label = np.random.choice(possible_labels)
        dataset.labels[idx] = poisoned_label

    return poison_indices