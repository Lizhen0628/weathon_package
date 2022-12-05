# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 10:22
# @Author  : LiZhen
# @FileName: tokenizer.py
# @github  : https://github.com/Lizhen0628
# @Description:


from abc import ABC,abstractmethod
import numpy as np
from transformers import BertTokenizer, AutoTokenizer
from weathon.nlp.base.vocab import BaseVocab
from typing import Union, List, Set


class BaseTokenizer(ABC):
    """
    分词器基类
    """

    def __init__(self, vocab: Union[BaseVocab, BertTokenizer, AutoTokenizer], max_seq_len: int):
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def tokenize(self, text: str) -> List[str]:
        return self.vocab.tokenize(text)

    def pad_and_truncate(self, sequence: List[int], maxlen: int, dtype: str = 'int64', padding: str = 'post',
                         truncating='post', value=0) -> np.ndarray:
        seq = (np.ones(maxlen) * value).astype(dtype)
        trunc = sequence[:maxlen] if truncating == 'post' else sequence[-maxlen:]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            seq[:len(trunc)] = trunc
        else:
            seq[-len(trunc)] = trunc
        return seq

    @abstractmethod
    def sequence_to_ids(self,sequence, reverse=False, padding="post", truncating="post",**kwargs):
        raise NotImplementedError("sequence_to_ids not implement")