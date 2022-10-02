import re
import pickle
import jieba

from zhon.hanzi import punctuation
from collections import Counter
from weathon.nlp.base import BaseVocab
from typing import Union, List, Set


class WordVocab(BaseVocab):

    def __init__(self, initial_tokens: Union[List[str], Set[str]] = None, vocab_size=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.initial_tokens = self.initial_vocab(initial_tokens) if initial_tokens is not None else []

        self.initial_tokens.insert(0, self.unk_token)
        self.initial_tokens.insert(0, self.pad_token)

        for token in self.initial_tokens:
            self.add(token)

    def add(self, token: str, cnt=1):
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
            self.vocab_size += 1

        return idx

    def initial_vocab(self, initial_tokens: Union[List[str], Set[str]]) -> List[str]:
        counter = Counter(initial_tokens)
        if self.vocab_size:
            vocab_size = self.vocab_size - 2
        else:
            vocab_size = len(counter)
        count_pairs = counter.most_common(vocab_size)

        tokens, _ = list(zip(*count_pairs))
        return list(tokens)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.get_id(term) for term in tokens]

    def recover_from_ids(self, ids, stop_id=None):
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens

    def get_id(self, token: str) -> int:
        return self.token2id[token] if token in self.token2id else self.token2id[self.unk_token]

    def get_token(self, idx: int) -> str:
        return self.id2token[idx] if idx in self.id2token else self.unk_token

    @classmethod
    def tokenize(cls, text: str, stop_words: Union[List[str], Set[str]] = None, mode: str = 'jieba',
                 lower: bool = True):
        text = re.sub(r'[%s]+' % punctuation, ' ', text)
        if lower:
            text = text.lower()

        if mode == 'jieba':
            if not hasattr(cls, 'word_tokenize'):
                cls.word_tokenize = lambda x: jieba.lcut(x)
            tokens = cls.word_tokenize(text)
        elif mode == 'pkuseg':
            if not hasattr(cls, 'word_tokenize'):
                cls.word_tokenizer = pkuseg.pkuseg(model_name='medicine')
                cls.word_tokenize = lambda x: cls.word_tokenizer.cut(x)
            tokens = cls.word_tokenize(text)
        else:
            # To Do: add eorr message
            pass

        if stop_words:
            tokens = filter(lambda w: w not in stop_words, tokens)
        return list(tokens)
