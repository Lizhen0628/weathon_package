# -*- coding: utf-8 -*-
# @Time    : 2022/10/4 10:19
# @Author  : LiZhen
# @FileName: sequence_classification.py
# @github  : https://github.com/Lizhen0628
# @Description:

import time
import torch
import warnings


from torch.utils.data import DataLoader
from weathon.nlp.base import BaseTask


class SequenceClassificationTask(BaseTask):
    """
    序列分类任务的基类

    Args:
        module: 深度学习模型
        optimizer: 训练模型使用的优化器名或者优化器对象
        loss_function: 训练模型使用的损失函数名或损失函数对象
        class_num (:obj:`int` or :obj:`None`, optional, defaults to None): 标签数目
        scheduler (:obj:`class`, optional, defaults to None): scheduler对象
        n_gpu (:obj:`int`, optional, defaults to 1): GPU数目
        device (:obj:`class`, optional, defaults to None): torch.device对象，当device为None时，会自动检测是否有GPU
        cuda_device (:obj:`int`, optional, defaults to 0): GPU编号，当device为None时，根据cuda_device设置device
        ema_decay (:obj:`int` or :obj:`None`, optional, defaults to None): EMA的加权系数
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):
        super(SequenceClassificationTask, self).__init__(*args, **kwargs)
        if hasattr(self.model, 'task') is False:
            self.model.task = 'SequenceLevel'

