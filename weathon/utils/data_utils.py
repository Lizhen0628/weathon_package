# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 21:37
# @Author  : LiZhen
# @FileName: data_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

from typing import Tuple
from torch.utils.data import random_split


class DataUtils:

    @staticmethod
    def random_split(dataset, val_proportion: float = 0.25) -> Tuple:
        """
        按照val_proportion比例，随机切分dataset
        Args:
            dataset: 原数据集
            val_proportion: 验证集占比
        Returns
            (训练集,验证集)
        """
        valid_dataset_size = int(len(dataset) * val_proportion)
        train_dataset_size = len(dataset) - valid_dataset_size
        train_dataset, valid_dataset = random_split(dataset=dataset, lengths=[train_dataset_size, valid_dataset_size])
        train_indices, valid_indices = train_dataset.indices, valid_dataset.indices
        train_dataset, valid_dataset = train_dataset.dataset, valid_dataset.dataset
        t_dataset = [train_dataset.dataset[idx] for idx in train_indices]
        v_dataset = [valid_dataset.dataset[idx] for idx in valid_indices]
        train_dataset.dataset = t_dataset
        valid_dataset.dataset = v_dataset
        return train_dataset, valid_dataset
