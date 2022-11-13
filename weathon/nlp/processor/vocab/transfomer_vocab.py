import jieba

from typing import List
from transformers import BertTokenizer


class TransfomerWithBlankVocab(BertTokenizer):
    def tokenize(self, text: str, **kwargs) -> List[str]:
        tokens = []
        for span_ in text.split():
            tokens += self._tokenize(span_)
            tokens += [' ']
        return tokens[:-1]


class RoFormerVocab(BertTokenizer):

    def tokenize(self, text: str, **kwargs):
        return list(jieba.cut(text))
