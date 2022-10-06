# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 11:58
# @Author  : LiZhen
# @FileName: model_ensemble.py
# @github  : https://github.com/Lizhen0628
# @Description:

import torch
from typing import List


class ModelEnsemble:

    @staticmethod
    def save_avg_model(model_list_path, avg_model_path):
        avg_model = None
        for i, model_path in enumerate(model_list_path):
            model = torch.load(model_path)
            if i == 0:
                avg_model = model
            else:
                for k, _ in avg_model.items():
                    avg_model[k].mul_(i).add_(model[k]).div_(i + 1)
        torch.save(avg_model, avg_model_path)
