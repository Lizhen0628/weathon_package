# -*- coding: utf-8 -*-
# @Time    : 2022/10/9 21:27
# @Author  : LiZhen
# @FileName: environment_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

import os
import torch
import random
import numpy as np


class EnvironmentUtils:

    @staticmethod
    def set_environ():
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    @staticmethod
    def set_seed(seed: int = 7):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
