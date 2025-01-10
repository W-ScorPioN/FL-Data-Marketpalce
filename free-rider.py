#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

"""
使用前提：
    1. 保存本地模型；
功能：
    1. 向模型添加噪声来仿真free-rider方法，通过设置noise_level控制噪声等级
"""

import torch
import copy
import gc

from utils.info import load_model_weights
from utils.options import args_parser
import time
import logging

# 向模型添加噪声
def add_noise_to_model(model, noise_level):
    noisy_model = copy.deepcopy(model)  # 复制原始模型
    with torch.no_grad():
        for param in noisy_model.parameters():
            noise = torch.randn_like(param) * noise_level
            param.add_(noise)
    return noisy_model

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    # parse args
    start = time.time()
    args = args_parser()
    logging.basicConfig(
        filename=f'./save/cosine_similarity/{args.dataset}_{args.model}_{str(args.epochs)}_{str(args.num_users)}_{args.model}_{str(args.noise_level)}_{str(args.frac)}_{str(args.bs)}_{args.task_id}.log',
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 初始化变量
    local_models_path = ["./save/model/xxx1.pth", "./save/model/xxx2.pth"]  # 本地模型路径，可以优化一下

    # 生成一组新的模型
    model_variants = []

    for i in range(len(local_models_path)):
        local_model = load_model_weights(local_models_path[i], args.device)
        new_model = add_noise_to_model(local_model, args.noise_level)
        model_variants.append(new_model)


    # 打印每个模型的参数以观察差异
    for i, model in enumerate(model_variants):
        logging.info(f"\nParameters of model variant {i+1}:")
        for param in model.parameters():
            logging.info(param)