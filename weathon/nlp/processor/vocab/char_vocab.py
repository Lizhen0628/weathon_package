import re

from zhon.hanzi import punctuation
from collections import Counter
from weathon.nlp.base import BaseVocab
from typing import Union, List, Set


class CharVocab(BaseVocab):

    def __init__(self,
                 initial_tokens: Union[List[str], Set[str]] = None,
                 vocab_size: int = 0,
                 tokenize_mode: str = 'zh',
                 edge_num=None,
                 adj_matrix=None,
                 edge_weight=None,
                 window_size=None):

        super().__init__()
        self.edge_num = edge_num
        self.adj_matrix = adj_matrix
        self.edge_weight = edge_weight
        self.window_size = window_size

        self.tokenize_mode = tokenize_mode

        self.vocab_size = vocab_size

        self.initial_tokens = self.initial_vocab(initial_tokens) if initial_tokens is not None else []

        self.initial_tokens.insert(0, self.unk_token)
        self.initial_tokens.insert(0, self.pad_token)

        for token in self.initial_tokens:
            self.add(token)

    def initial_vocab(self, initial_tokens: Union[List[str], Set[str]]) -> List[str]:
        counter = Counter(initial_tokens)
        vocab_size = self.vocab_size - 2 if self.vocab_size else len(counter)
        count_pairs = counter.most_common(vocab_size)
        tokens, _ = list(zip(*count_pairs))
        return list(tokens)

    def add(self, token: str, cnt=1) -> int:
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
            self.vocab_size += 1
        return idx

    def get_id(self, token: str) -> int:
        return self.token2id[token] if token in self.token2id else self.token2id[self.unk_token]

    def get_token(self, idx: int) -> str:
        return self.id2token[idx] if idx in self.id2token else self.unk_token

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.get_id(term) for term in tokens]

    def recover_from_ids(self, ids: List[int], stop_id=None) -> List[str]:
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens

    def tokenize(self, text: str, stop_words: Union[List[str], Set[str]] = None, lower: bool = True):
        if self.tokenize_mode == 'zh':
            return CharVocab.zh_tokenize(text, stop_words, lower)
        elif self.tokenize_mode == 'en':
            return CharVocab.en_tokenize(text, stop_words, lower)
        else:
            raise ValueError('没有该分词模式')

    @classmethod
    def zh_tokenize(cls, text: str, stop_words: Union[List[str], Set[str]] = None, lower: bool = True):
        text = re.sub(r'[%s]+' % punctuation, '', text)
        if lower:
            text = text.lower()
        tokens = [token_ for token_ in text]

        if stop_words:
            tokens = filter(lambda w: w not in stop_words, tokens)

        return list(tokens)

    @classmethod
    def en_tokenize(cls, text: str, stop_words: Union[List[str], Set[str]] = None, lower: bool = True):
        if lower:
            text = text.lower()
        tokens = text.split()
        if stop_words:
            tokens = filter(lambda w: w not in stop_words, tokens)
        return list(tokens)
