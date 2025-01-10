#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

"""
使用前提：
    1. 保存本地模型；
    2. 保存全局模型；
功能：
    1. 基于余弦相似度、欧氏距离、曼哈顿距离三种方法构建了一个聚合器，计算每个客户端对全局模型的贡献度；
"""
import gc
from utils.options import args_parser
import time
import logging
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class ModelAggregator:
    def __init__(self, metric='cosine', n_clusters=2, linkage='average'):
        self.metric = metric
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, local_models, global_model):
        # 加载模型参数并展平
        global_params = self._flatten_model_params(global_model)
        local_params = [self._flatten_model_params(model) for model in local_models]

        # 计算相似度或距离
        if self.metric == 'cosine':
            matrix = cosine_similarity([global_params] + local_params)
        elif self.metric == 'euclidean':
            matrix = pairwise_distances([global_params] + local_params, metric='euclidean')
        elif self.metric == 'manhattan':
            matrix = pairwise_distances([global_params] + local_params, metric='manhattan')
        else:
            raise ValueError("Unsupported metric")

        # 提取本地模型之间的相似度或距离矩阵
        local_matrix = matrix[1:, 1:] if self.metric == 'cosine' else matrix

        # 聚类
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed' if self.metric == 'cosine' else 'euclidean',
            linkage=self.linkage
        )

        # 使用1 - 相似度作为距离（如果是余弦相似度）
        data_to_cluster = 1 - local_matrix if self.metric == 'cosine' else local_matrix
        labels = clustering.fit_predict(data_to_cluster)
        return labels

    def _flatten_model_params(self, model):
        # 展平模型参数
        state_dict = model.state_dict() if isinstance(model, torch.nn.Module) else torch.load(model)
        params = np.concatenate([param.flatten().numpy() for param in state_dict.values()])
        return params


def plot_clusters(data, labels, n_component, title='Cluster Plot'):
    """
    使用t-SNE对数据降维并绘制聚类结果。

    :param data: 原始高维数据。
    :param labels: 聚类标签。
    :param title: 图的标题。
    """
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=n_component, random_state=42)
    reduced_data = tsne.fit_transform(data)

    # 绘制集群
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.colorbar(scatter)
    plt.xlabel("t-SNE Component 1") # 根据需要自行修改
    plt.ylabel("t-SNE Component 2") # 根据需要自行修改
    plt.show()

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    # parse args
    start = time.time()
    args = args_parser()
    logging.basicConfig(
        filename=f'./save/cosine_similarity/{args.dataset}_{args.model}_{str(args.epochs)}_{str(args.num_users)}_{args.model}_{str(args.iid)}_{str(args.frac)}_{str(args.bs)}_{args.task_id}.log',
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 初始化变量
    global_model_path = "./save/model/xxx.pth"  # 全局模型路径，可以优化一下
    local_models_path = ["./save/model/xxx1.pth", "./save/model/xxx2.pth"]  # 本地模型路径，可以优化一下

    aggregator = ModelAggregator(metric=args.similarity, n_clusters=args.n_clusters, linkage='average')
    labels = aggregator.fit_predict(local_models_path, global_model_path)

    # 将所有模型参数展平并组合成一个数据矩阵
    global_params = aggregator._flatten_model_params(global_model_path)
    local_params = [aggregator._flatten_model_params(model) for model in local_models_path]
    all_params = np.vstack([global_params] + local_params)

    # 绘制聚类结果
    plot_clusters(all_params, labels, args.n_clusters, title='Model Clustering Result')

    logging.info(f"Cluster labels: {labels}")

