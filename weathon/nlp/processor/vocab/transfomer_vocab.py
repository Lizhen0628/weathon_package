import jieba

from typing import Union,List,Set
from transformers import BertTokenizer


class TransfomerWithBlankVocab(BertTokenizer):
    def tokenize(self, text, **kwargs) -> List[str]:
        tokens = []
        for span_ in text.split():
            tokens += self._tokenize(span_)
            tokens += [' ']
        return tokens[:-1]


class RoFormerVocab(BertTokenizer):

    def tokenize(self, text):
        return list(jieba.cut(text))
