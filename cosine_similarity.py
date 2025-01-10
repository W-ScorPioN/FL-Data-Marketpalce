#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

"""
使用前提：
    1. 保存本地模型；
    2. 保存全局模型；
功能：
    1. 基于余弦相似度方法，计算每个客户端对全局模型的贡献度；
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import copy
import gc
from utils.options import args_parser
from utils.info import load_model_weights
import time
import logging


# 计算余弦相似度
def calculate_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# 计算每个本地模型对全局模型的贡献度
def calculate_contributions(global_model_path, local_model_paths, device):
    global_weights = load_model_weights(global_model_path, device)
    contributions = []

    for local_model_path in local_model_paths:
        local_weights = load_model_weights(local_model_path, device)
        similarity = calculate_cosine_similarity(local_weights, global_weights)
        contributions.append(similarity)

    return contributions


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    # parse args
    start = time.time()
    args = args_parser()
    logging.basicConfig(filename=f'./save/cosine_similarity/{args.dataset}_{args.model}_{str(args.epochs)}_{str(args.num_users)}_{args.model}_{str(args.iid)}_{str(args.frac)}_{str(args.bs)}_{args.task_id}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    # 初始化变量
    global_model_path = "./save/model/xxx.pth" # 全局模型路径，可以优化一下
    local_models_path = ["./save/model/xxx1.pth", "./save/model/xxx2.pth"] # 本地模型路径，可以优化一下

    # 计算用户本地模型对全局模型的贡献
    start = time.time()
    contributions = calculate_contributions(global_model_path, local_models_path, args.device)
    end = time.time()
    logging.info(f"Time cost of Cosine Similarity is: {end - start}")

    for i, contribution in enumerate(contributions):
        logging.info(f"Client {i+1} contribution: {contribution:.4f}")

