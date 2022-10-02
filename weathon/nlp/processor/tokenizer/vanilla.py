from typing import Union, List, Set
from weathon.nlp.base import BaseTokenizer, BaseVocab
from transformers import BertTokenizer, AutoTokenizer


class VanillaTokenizer(BaseTokenizer):
    """
    Reference:
        [1] https://github.com/dasiki/https-github.com-ami66-ChineseTextClassifier
    """

    def __init__(self, vocab: Union[BaseVocab, BertTokenizer, AutoTokenizer], max_seq_len: int):
        super(VanillaTokenizer, self).__init__(vocab, max_seq_len)
        self.tokenizer_type = 'vanilla'

    def sequence_to_ids(self, sequence, reverse=False, padding='post', truncating='post'):
        if type(sequence) == str:
            sequence = self.tokenize(sequence)

        sequence = self.vocab.convert_tokens_to_ids(sequence)
        if len(sequence) == 0:
            sequence = [0]

        if reverse:
            sequence = sequence[::-1]

        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
