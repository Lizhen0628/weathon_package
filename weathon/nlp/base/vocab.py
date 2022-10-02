# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 09:24
# @Author  : LiZhen
# @FileName: vocab.py
# @github  : https://github.com/Lizhen0628
# @Description:

import abc
import json
from typing import List


class BaseVocab(object, metaclass=abc.ABCMeta):

    def __init__(self):
        self.id2token = {}
        self.token2id = {}

        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'

    @abc.abstractmethod
    def add(self, token: str, cnt=1) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_id(self, token: str) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_token(self,idx:int) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def recover_id2token(self):
        return {idx: token for token, idx in self.token2id.items()}

    def save(self, output_path: str = './token2id.json') -> None:
        with open(output_path, mode='w', encoding='utf8') as f:
            json.dump(obj=self.token2id, fp=f, ensure_ascii=False)

    def load(self, save_path='./token2id.json') -> None:
        with open(save_path, mode='r', encoding='utf8') as f:
            self.token2id = json.load(fp=f)
        self.id2token = self.recover_id2token()
