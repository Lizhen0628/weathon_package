# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 11:55
# @Author  : LiZhen
# @FileName: seed_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

import random
import os
import numpy as np
import torch


class Seed:

    @staticmethod
    def set_seed( seed=7):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
