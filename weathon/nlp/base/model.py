# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 15:40
# @Author  : LiZhen
# @FileName: basemodel.py
# @github  : https://github.com/Lizhen0628
# @Description:


import torch
import time
import numpy as np
from pathlib import Path
from typing import Union


class BaseModel(torch.nn.Module):
    """
    封装了nn.Module，主要是提供了save和load两个方法

    Attributes:
        model_name (str): 模型名称
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path: Union[str, Path]):
        """可加载指定路径的模型"""
        self.load_state_dict(torch.load(path))

    def save(self, model_path=None):
        """保存模型，默认使用“模型名字+时间”作为文件名"""
        if model_path is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            model_path = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), model_path)
        return model_path

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class Flat(torch.nn.Module):
    """Flat类，把输入reshape成（batch_size,dim_length）"""

    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
