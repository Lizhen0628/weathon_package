# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 09:24
# @Author  : LiZhen
# @FileName: vocab.py
# @github  : https://github.com/Lizhen0628
# @Description:

import json
from typing import List
from abc import ABC, abstractmethod


class BaseVocab(ABC):

    def __init__(self):
        self.id2token = {}
        self.token2id = {}

        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'

    @abstractmethod
    def add(self, token: str) -> int:
        """
        向词表中添加token，并返回token在词表中对应的index
        Args:
            token:
        Returns:
            token在词表中对应的id
        """
        raise NotImplementedError

    @abstractmethod
    def get_id(self, token: str) -> int:
        """
        获取token对应的token_id，如果token在词表中不存在，返回unk_token对应的id
        Args:
            token: 输入token

        Returns:返回token对应的token_id。

        """
        raise NotImplementedError

    @abstractmethod
    def get_token(self, idx: int) -> str:
        """
        根据idx返回对应的token
        Args:
            idx:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
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
