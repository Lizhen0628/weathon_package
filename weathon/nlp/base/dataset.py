# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 13:57
# @Author  : LiZhen
# @FileName: dataset.py
# @github  : https://github.com/Lizhen0628
# @Description:

import json
import copy
import codecs
import pandas as pd
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, List
from collections import defaultdict
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
from weathon.utils import FileUtils


# TODO:dataset split


class BaseDataset(Dataset):
    """
    Dataset基类

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def __init__(self,
                 data: Union[DataFrame, str, Path],  # 数据或者数据地址
                 categories=None,  # 数据类别
                 is_retain_df: bool = False,  # 是否将DataFrame格式的原始数据复制到属性retain_df中
                 is_retain_dataset: bool = False,  # 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
                 is_test: bool = False
                 ):

        self.is_test = is_test
        self.is_retain_df = is_retain_df
        self.is_retain_dataset = is_retain_dataset

        if isinstance(data, DataFrame):
            if 'label' in data.columns:
                data.loc[:]['label'] = data.loc[:]['label'].apply(lambda x: str(x))

            self.df = data if is_retain_df else None
            self.dataset = self._convert_to_dataset(data)
        else:
            self.dataset = self._load_dataset(data)

        self.retain_dataset = copy.deepcopy(self.dataset) if is_retain_dataset else None

        self.categories = categories if categories else self._get_categories()

        if self.categories is not None:
            self.cat2id = dict(zip(self.categories, range(len(self.categories))))
            self.id2cat = dict(zip(range(len(self.categories)), self.categories))

            self.class_num = len(self.cat2id)

    @abstractmethod
    def _get_categories(self):
        raise NotImplementedError("_get_categories method not implement")

    @abstractmethod
    def _convert_to_dataset(self, data_df):
        raise NotImplementedError("_convert_to_dataset method not implement")

    def _load_dataset(self, data_path: Path):
        """
        加载数据集

        Args:
            data_path (:obj:`string`): 数据地址
        """  # noqa: ignore flake8"

        data_df = self._read_data(data_path)
        self.df = data_df if self.is_retain_df else None
        return self._convert_to_dataset(data_df)

    def _read_data(self, data_path: Path):
        """
        读取所需数据
        Args:
            data_path (:obj:`string`): 数据地址
        """  # noqa: ignore flake8"

        if data_path.suffix == '.csv':
            data_df = pd.read_csv(data_path, dtype={'label': str})
        elif data_path.suffix == '.json' or data_path.suffix == '.jsonl':
            data_df = FileUtils.read_json(data_path)
        elif data_path.suffix == '.tsv':
            data_df = pd.read_csv(data_path, sep='\t', dtype={'label': str})
        elif data_path.suffix == '.txt':
            data_df = pd.read_csv(data_path, sep='\t', dtype={'label': str})
        else:
            raise ValueError("The data format does not exist")

        return data_df

    def read_line_json(self, data_path: Path):
        """
        读取所需数据

        Args:
            data_path (:obj:`string`): 数据所在路径
            skiprows (:obj:`int`, defaults to -1): 读取跳过指定行数，默认为不跳过
        """
        datasets = []

        with data_path.open(mode="r", encoding="utf8") as reader:
            for line in reader:
                json_line = json.loads(line)
                datasets.append(json_line)
                # tokens = line['text']
                # label = line['label']
                # datasets.append({'text': tokens.strip(), 'label': label})
        return pd.DataFrame(datasets)

    def convert_to_ids(self, tokenizer):
        """
        将文本转化成id的形式
        Args:
            tokenizer: 编码器
        """
        if tokenizer.tokenizer_type == 'vanilla':
            features = self._convert_to_vanilla_ids(tokenizer)
        elif tokenizer.tokenizer_type == 'transformer':
            features = self._convert_to_transfomer_ids(tokenizer)
        elif tokenizer.tokenizer_type == 'customized':
            features = self._convert_to_customized_ids(tokenizer)
        else:
            raise ValueError("The tokenizer type does not exist")

        self.dataset = features

    def _convert_to_transfomer_ids(self, bert_tokenizer):
        pass

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):
        pass

    def _convert_to_customized_ids(self, customized_tokenizer):
        pass

    @property
    def dataset_cols(self) -> List[str]:
        return list(self.dataset[0].keys())

    @property
    def to_device_cols(self) -> List[str]:
        return list(self.dataset[0].keys())

    @property
    def sample_num(self):
        return len(self.dataset)

    @property
    def dataset_analysis(self):

        _result = defaultdict(list)
        for _row in self.dataset:
            for _col in self.dataset_cols:
                if type(_row[_col]) == str:
                    _result[_col].append(len(_row[_col]))

        _report = pd.DataFrame(_result).describe()

        return _report

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class TokenClassificationDataset(BaseDataset, ABC):
    """
    用于字符分类任务的Dataset

    # pandas dataframe的columns必选包含"text"和"label"
    # text列为文本
    # label列为列表形式，列表中每个元素是如下组织的字典
    # {'start_idx': 实体首字符在文本的位置, 'end_idx': 实体尾字符在文本的位置, 'type': 实体类型标签, 'entity': 实体}
    """

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text'] = data_df['text'].apply(lambda x: x.strip())
        if not self.is_test:
            data_df['label'] = data_df['label'].apply(lambda x: eval(x) if type(x) == str else x)

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        return dataset
