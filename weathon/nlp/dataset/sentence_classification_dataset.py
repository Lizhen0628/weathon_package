# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 16:47
# @Author  : LiZhen
# @FileName: sentence_classification_dataset.py.py
# @github  : https://github.com/Lizhen0628
# @Description:

import copy
import numpy as np
from typing import List, Dict
from weathon.nlp.base import BaseDataset
from weathon.nlp.processor.tokenizer import SentenceTokenizer


class SentenceClassificationDataset(BaseDataset):
    """
    用于序列分类任务的Dataset
    """

    def _get_categories(self) -> List[str]:
        return sorted(list(set([data['label'] for data in self.dataset])))

    def _convert_to_dataset(self, data_df) -> List[Dict]:

        dataset = []
        data_df.loc[:]['text'] = data_df.loc[:]['text'].apply(lambda x: str(x).lower().strip())

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        return dataset

    def _convert_to_transfomer_ids(self, bert_tokenizer: SentenceTokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = bert_tokenizer.sequence_to_ids(row_['text'])
            input_ids, input_mask, segment_ids = input_ids
            feature = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features

    def _convert_to_vanilla_ids(self, vanilla_tokenizer) -> List[Dict]:

        features = []
        for (index_, row_) in enumerate(self.dataset):
            tokens = vanilla_tokenizer.tokenize(row_['text'])
            length = len(tokens)
            input_ids = vanilla_tokenizer.sequence_to_ids(tokens)

            feature = {
                'input_ids': input_ids,
                'length': length if length < vanilla_tokenizer.max_seq_len else vanilla_tokenizer.max_seq_len
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features


class PairMergeSentenceClassificationDataset(BaseDataset):
    """
    用于句子对合并后进行序列分类任务的Dataset，例如BERT分类任务
    """

    def _get_categories(self):
        return sorted(list(set([data['label'] for data in self.dataset])))

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text_a'] = data_df['text_a'].apply(lambda x: x.lower().strip())
        data_df['text_b'] = data_df['text_b'].apply(lambda x: x.lower().strip())

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        return dataset

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = bert_tokenizer.sequence_to_ids(row_['text_a'], row_['text_b'])

            input_ids, input_mask, segment_ids = input_ids

            feature = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features


class TwinTowersSentenceClassificationDataset(BaseDataset):
    """
    用于双塔序列分类任务的Dataset，即句子对不组合，分开输入模型
    """

    def _get_categories(self):
        return sorted(list(set([data['label'] for data in self.dataset])))

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text_a'] = data_df['text_a'].apply(lambda x: x.lower().strip())
        data_df['text_b'] = data_df['text_b'].apply(lambda x: x.lower().strip())

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        return dataset

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):

            input_ids_a = bert_tokenizer.sequence_to_ids(row_['text_a'])
            input_ids_b = bert_tokenizer.sequence_to_ids(row_['text_b'])

            input_ids_a, input_mask_a, segment_ids_a = input_ids_a
            input_ids_b, input_mask_b, segment_ids_b = input_ids_b

            feature = {
                'input_ids_a': input_ids_a,
                'attention_mask_a': input_mask_a,
                'token_type_ids_a': segment_ids_a,
                'input_ids_b': input_ids_b,
                'attention_mask_b': input_mask_b,
                'token_type_ids_b': segment_ids_b
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):

            input_ids_a = vanilla_tokenizer.sequence_to_ids(row_['text_a'])
            input_ids_b = vanilla_tokenizer.sequence_to_ids(row_['text_b'])

            feature = {
                'input_ids_a': input_ids_a,
                'input_ids_b': input_ids_b
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features


class TCDataset(SentenceClassificationDataset):
    """
    用于文本分类任务的Dataset

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):
        super(TCDataset, self).__init__(*args, **kwargs)


class TMDataset(PairMergeSentenceClassificationDataset):
    """
    用于文本匹配任务的Dataset

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):
        super(TMDataset, self).__init__(*args, **kwargs)


class PromptDataset(SentenceClassificationDataset):
    """
    用于使用prompt的自然语言处理任务的Dataset

    Args:
        data (DataFrame or string): 数据或者数据地址
        prompt (list): 所使用的prompt, 如["是", "[MASK]"]
        prompt_mode (string): 
            prompt放置在文本中的方式
            有postfix和prefix两种, postfix表示text + prompt, prefix表示prompt + text
            默认值为"postfix"
        prefix_special_token_num (int): 使用前缀特殊字符的数量, 如"[CLS]", 默认值为1
        categories (list, optional): 数据类别, 默认值为None
        is_retain_df (bool, optional): 是否将DataFrame格式的原始数据复制到属性retain_df中, 默认值为False
        is_retain_dataset (bool, optional): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中, 默认值为False
        is_train (bool, optional): 数据集是否为训练集数据, 默认值为True
        is_test (bool, optional): 数据集是否为测试集数据, 默认值为False
    """  # noqa: ignore flake8"

    def __init__(
            self,
            *args,
            prompt,
            prompt_mode='postfix',
            prefix_special_token_num=1,
            **kwargs
    ):
        super(PromptDataset, self).__init__(*args, **kwargs)

        self.prompt = prompt
        self.mask_lm_label_size = self.prompt.count("[MASK]")

        self.prompt_mode = prompt_mode
        self.prefix_special_token_num = prefix_special_token_num

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []

        for (index_, row_) in enumerate(self.dataset):

            seq = bert_tokenizer.tokenize(row_['text'])

            if self.prompt_mode == 'postfix':
                start_mask_position = len(seq) + self.prefix_special_token_num + self.prompt.index("[MASK]")
            else:
                start_mask_position = self.prefix_special_token_num + self.prompt.index("[MASK]")

            mask_position = [
                start_mask_position + index
                for index in range(self.mask_lm_label_size)
            ]

            input_ids = bert_tokenizer.sequence_to_ids(row_['text'], self.prompt, self.prompt_mode)

            input_ids, input_mask, segment_ids = input_ids

            feature = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'mask_position': np.array(mask_position, dtype='int64')
            }

            if not self.is_test:
                mask_lm_label = bert_tokenizer.vocab.convert_tokens_to_ids(bert_tokenizer.tokenize(row_['label']))

                feature['label_ids'] = np.array(mask_lm_label, dtype='int64')

            features.append(feature)

        return features


class TextLevelGCNDataset(SentenceClassificationDataset):

    def convert_to_ids(self, tokenizer):
        """
        将文本转化成id的形式

        :param tokenizer:
        """
        if tokenizer.tokenizer_type == 'graph':
            features = self._convert_to_graph_ids(tokenizer)
        else:
            raise ValueError("The tokenizer type does not exist")

        if self.is_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)

        self.dataset = features

    def _convert_to_graph_ids(self, graph_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            node_ids, edge_ids, sub_graph = graph_tokenizer.sequence_to_graph(row_['text'])

            feature = {
                'node_ids': node_ids,
                'edge_ids': edge_ids,
                'sub_graph': sub_graph
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features
