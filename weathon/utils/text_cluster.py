# -*- coding: utf-8 -*-
# @Time    : 2022/10/6 13:51
# @Author  : LiZhen
# @FileName: text_cluster.py
# @github  : https://github.com/Lizhen0628
# @Description:
import math
import random

from collections import Counter, OrderedDict


class TextCluster:

    def elu_distance(self, a, b):
        """计算两点之间的欧氏距离并返回

        :param a: list of float
        :param b: list of float
        :return: float
        """

        x = sum([pow((a_ - b_), 2) for a_, b_ in zip(a, b)])
        return math.sqrt(x)

    def count_features(self, corpus, tokenizer=list):
        """词频特征

        :param corpus: list of str
        :param tokenizer: function for tokenize, default is `jiagu.cut`
        :return:
            features: list of list of float
            names: list of str

        example:
        corpus = ["判断unicode是否是汉字，数字，英文，或者其他字符。", "全角符号转半角符号。"]
        X, names = count_features(corpus)
        """
        tokens = [tokenizer(x) for x in corpus]
        vocab = [x[0] for x in Counter([x for s in tokens for x in s]).most_common()]

        features = []
        for sent in tokens:
            counter = Counter(sent)
            feature = [counter.get(x, 0) for x in vocab]
            features.append(feature)

        return features, vocab

    def tfidf_features(self, corpus, tokenizer=list):
        """文本的 tfidf 特征

        :param corpus: list of str
        :param tokenizer: function for tokenize, default is `jiagu.cut`
        :return:
            features: list of list of float
            names: list of str

        example:
        corpus = ["判断unicode是否是汉字。", "全角符号转半角符号。", "一些基于自然语言处理的预处理过程也会在本文中出现。"]
        X, names = tfidf_features(corpus, tokenizer=jieba.cut)
        """
        tokens = [tokenizer(x) for x in corpus]
        vocab = [x[0] for x in Counter([x for s in tokens for x in s]).most_common()]

        idf_dict = dict()
        total_doc = len(corpus)
        for word in vocab:
            num = sum([1 if (word in s) else 0 for s in corpus])
            if num == total_doc:
                idf = math.log(total_doc / num)
            else:
                idf = math.log(total_doc / (num + 1))
            idf_dict[word] = idf

        features = []
        for sent in tokens:
            counter = Counter(sent)
            feature = [counter.get(x, 0) / len(sent) * idf_dict.get(x, 0) for x in vocab]
            features.append(feature)

        return features, vocab

    def text_cluster(self, docs, features_method='tfidf', method="dbscan",
                     k=3, max_iter=100, eps=0.5, min_pts=2, tokenizer=list):
        """文本聚类，目前支持 K-Means 和 DBSCAN 两种方法

        :param features_method: str
            提取文本特征的方法，目前支持 tfidf 和 count 两种。
        :param docs: list of str
            输入的文本列表，如 ['k-means', 'dbscan']
        :param method: str
            指定使用的方法，默认为 k-means，可选 'k-means', 'dbscan'
        :param k: int
            k-means 参数，类簇数量
        :param max_iter: int
            k-means 参数，最大迭代次数，确保模型不收敛的情况下可以退出循环
        :param eps: float
            dbscan 参数，邻域距离
        :param min_pts:
            dbscan 参数，核心对象中的最少样本数量
        :return: dict
            聚类结果
        """
        if features_method == 'tfidf':
            features, names = self.tfidf_features(docs, tokenizer)
        elif features_method == 'count':
            features, names = self.count_features(docs, tokenizer)
        else:
            raise ValueError('features_method error')

        # feature to doc
        f2d = {k: v for k, v in zip(docs, features)}

        if method == 'k-means':
            km = KMeans(k=k, max_iter=max_iter)
            clusters = km.train(features)

        elif method == 'dbscan':
            ds = DBSCAN(eps=eps, min_pts=min_pts)
            clusters = ds.train(features)

        else:
            raise ValueError("method invalid, please use 'k-means' or 'dbscan'")

        clusters_out = {}

        for label, examples in clusters.items():
            c_docs = []
            for example in examples:
                doc = [d for d, f in f2d.items() if list(example) == f]
                c_docs.extend(doc)

            clusters_out[label] = list(set(c_docs))

        return clusters_out


class KMeans(object):
    def __init__(self, k, max_iter=100):
        """

        :param k: int
            类簇数量，如 k=5
        :param max_iter: int
            最大迭代次数，避免不收敛的情况出现导致无法退出循环，默认值为 max_iter=100
        """
        self.k = k
        self.max_iter = max_iter

        self.centroids = None  # list
        self.clusters = None  # OrderedDict

    def elu_distance(self, a, b):
        """计算两点之间的欧氏距离并返回

        :param a: list of float
        :param b: list of float
        :return: float
        """

        x = sum([pow((a_ - b_), 2) for a_, b_ in zip(a, b)])
        return math.sqrt(x)

    def _update_clusters(self, dataset):
        """
        对dataset中的每个点item, 计算item与centroids中k个中心的距离
        根据最小距离将item加入相应的簇中并返回簇类结果cluster
        """
        clusters = OrderedDict()
        centroids = self.centroids

        k = len(centroids)
        for item in dataset:
            a = item
            flag = -1
            min_dist = float("inf")

            for i in range(k):
                b = centroids[i]
                dist = self.elu_distance(a, b)
                if dist < min_dist:
                    min_dist = dist
                    flag = i

            if flag not in clusters.keys():
                clusters[flag] = []
            clusters[flag].append(item)

        self.clusters = clusters

    def _mean(self, features):
        res = []
        for i in range(len(features[0])):
            col = [x[i] for x in features]
            res.append(sum(col) / len(col))
        return res

    def _update_centroids(self):
        """根据簇类结果重新计算每个簇的中心，更新 centroids"""
        centroids = []
        for key in self.clusters.keys():
            centroid = self._mean(self.clusters[key])
            centroids.append(centroid)
        self.centroids = centroids

    def _quadratic_sum(self):
        """计算簇内样本与各自中心的距离，累计求和。

        sum_dist刻画簇内样本相似度, sum_dist越小则簇内样本相似度越高
        计算均方误差，该均方误差刻画了簇内样本相似度
        将簇类中各个点与质心的距离累计求和
        """
        centroids = self.centroids
        clusters = self.clusters

        sum_dist = 0.0
        for key in clusters.keys():
            a = centroids[key]
            dist = 0.0
            for item in clusters[key]:
                b = item
                dist += self.elu_distance(a, b)
            sum_dist += dist
        return sum_dist

    def train(self, X):
        """输入数据，完成 KMeans 聚类

        :param X: list of list
            输入数据特征，[n_samples, n_features]，如：[[0.36, 0.37], [0.483, 0.312]]
        :return: OrderedDict
        """
        # 随机选择 k 个 example 作为初始类簇均值向量
        self.centroids = random.sample(X, self.k)

        self._update_clusters(X)
        current_dist = self._quadratic_sum()
        old_dist = 0
        iter_i = 0

        while abs(current_dist - old_dist) >= 0.00001:
            self._update_centroids()
            self._update_clusters(X)
            old_dist = current_dist
            current_dist = self._quadratic_sum()

            iter_i += 1
            if iter_i > self.max_iter:
                break

        return self.clusters


class DBSCAN(object):
    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts

    def elu_distance(self, a, b):
        """计算两点之间的欧氏距离并返回

        :param a: list of float
        :param b: list of float
        :return: float
        """

        x = sum([pow((a_ - b_), 2) for a_, b_ in zip(a, b)])
        return math.sqrt(x)

    def _find_cores(self, X):
        """遍历样本集找出所有核心对象"""
        cores = set()
        for di in X:
            if len([dj for dj in X if self.elu_distance(di, dj) <= self.eps]) >= self.min_pts:
                cores.add(di)
        return cores

    def train(self, X):
        """输入数据，完成 KMeans 聚类

        :param X: list of tuple
            输入数据特征，[n_samples, n_features]，如：[[0.36, 0.37], [0.483, 0.312]]
        :return: OrderedDict
        """

        # 确定数据集中的全部核心对象集合
        X = [tuple(x) for x in X]
        cores = self._find_cores(X)
        not_visit = set(X)

        k = 0
        clusters = OrderedDict()
        while len(cores):
            not_visit_old = not_visit
            # 随机选取一个核心对象
            core = list(cores)[random.randint(0, len(cores) - 1)]
            not_visit = not_visit - set(core)

            # 查找所有密度可达的样本
            core_deque = [core]
            while len(core_deque):
                coreq = core_deque[0]
                coreq_neighborhood = [di for di in X if self.elu_distance(di, coreq) <= self.eps]

                # 若coreq为核心对象，则通过求交集方式将其邻域内未被访问过的样本找出
                if len(coreq_neighborhood) >= self.min_pts:
                    intersection = not_visit & set(coreq_neighborhood)
                    core_deque += list(intersection)
                    not_visit = not_visit - intersection

                core_deque.remove(coreq)
            cluster_k = not_visit_old - not_visit
            cores = cores - cluster_k
            clusters[k] = list(cluster_k)
            k += 1

        return clusters
