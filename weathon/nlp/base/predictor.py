# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 20:03
# @Author  : LiZhen
# @FileName: predictor.py
# @github  : https://github.com/Lizhen0628
# @Description:


import torch


class BasePredictor(object):
    """
    所有Predictor类的基类，封装Predictor类通用的方法和属性

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id
    ):
        self.module = module

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _convert_to_transfomer_ids(
            self,
            text
    ):
        input_ids = self.tokenizer.sequence_to_ids(text)
        input_ids, input_mask, segment_ids = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids
        }
        return features

    def _convert_to_vanilla_ids(
            self,
            text
    ):
        tokens = self.tokenizer.tokenize(text)
        length = len(tokens)
        input_ids = self.tokenizer.sequence_to_ids(tokens)

        features = {
            'input_ids': input_ids,
            'length': length if length < self.tokenizer.max_seq_len else self.tokenizer.max_seq_len,
        }
        return features

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text
    ):
        pass

    def _get_module_batch_inputs(
            self,
            features
    ):
        return {col: features[col].type(torch.long).to(self.device) for col in self.inputs_cols}

    def predict_batch(
            self,
            test_data
    ):
        pass

    def _threshold(
            self,
            x,
            threshold
    ):
        if x >= threshold:
            return 1
        return 0
