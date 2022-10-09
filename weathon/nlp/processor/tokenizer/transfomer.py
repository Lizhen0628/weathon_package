import warnings
import unicodedata
import transformers
import numpy as np

from typing import List, Union, Set, Tuple
from copy import deepcopy
from weathon.nlp.base import BaseVocab, BaseTokenizer
from transformers import BertTokenizer, AutoTokenizer


class TransfomerTokenizer(BaseTokenizer):
    """
    Transfomer文本编码器，用于对文本进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名，用于实现文本分词和ID化
        max_seq_len (:obj:`int`): 预设的文本最大长度
    """  # noqa: ignore flake8"

    def __init__(self, vocab: Union[BertTokenizer, AutoTokenizer, BaseVocab], max_seq_len: int):
        super().__init__(vocab, max_seq_len)
        if isinstance(vocab, str):
            # TODO: 改成由自定义的字典所决定
            try:
                vocab = AutoTokenizer.from_pretrained(vocab)
            except ValueError:
                vocab = BertTokenizer.from_pretrained(vocab)

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.additional_special_tokens = set()
        self.tokenizer_type = 'transformer'

    @staticmethod
    def _is_control(ch: str) -> bool:
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch: str) -> bool:
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @staticmethod
    def recover_bert_token(token: str) -> str:
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def get_token_mapping(self, text, tokens, is_mapping_index=True) -> List[str]:
        """给出原始的text和tokenize后的tokens的映射关系"""
        raw_text = deepcopy(text)
        text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            ch = unicodedata.normalize('NFD', ch)
            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            token = token.lower()
            if token == '[unk]' or token in self.additional_special_tokens:
                if is_mapping_index:
                    token_mapping.append(char_mapping[offset:offset + 1])
                else:
                    token_mapping.append(raw_text[offset:offset + 1])
                offset = offset + 1
            elif self._is_special(token):
                # 如果是[CLS]或者是[SEP]之类的词，则没有对应的映射
                token_mapping.append([])
            else:
                token = self.recover_bert_token(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                if is_mapping_index:
                    token_mapping.append(char_mapping[start:end])
                else:
                    token_mapping.append(raw_text[start:end])
                offset = end

        return token_mapping

    def sequence_to_ids(self, sequence_a: Union[str, List[str]], sequence_b: Union[str, List[str]] = None,
                        **kwargs) -> Tuple:
        return self.pair_to_ids(sequence_a, sequence_b, **kwargs) if sequence_b else self.sentence_to_ids(sequence_a,
                                                                                                          **kwargs)

    def sentence_to_ids(self, sequence: Union[str, List[str]], return_sequence_length: bool = False) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        if type(sequence) == str:
            sequence = self.tokenize(sequence)
        token_length = len(sequence)
        # 对超长序列进行截断
        if len(sequence) > self.max_seq_len - 2:
            sequence = sequence[0:(self.max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence + ['[SEP]']
        segment_ids = [0] * len(sequence)
        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')

        return (sequence, sequence_mask, segment_ids, token_length) if return_sequence_length else (
            sequence, sequence_mask, segment_ids)

    def pair_to_ids(self,
                    sequence_a: str,
                    sequence_b: str,
                    truncation_method: str = 'average',
                    return_sequence_length: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        if type(sequence_a) == str:
            sequence_a = self.tokenize(sequence_a)

        if type(sequence_b) == str:
            sequence_b = self.tokenize(sequence_b)

        sequence_length = (len(sequence_a), len(sequence_b))
        # 对超长序列进行截断
        if truncation_method == 'average':
            if len(sequence_a) > ((self.max_seq_len - 3) // 2):
                sequence_a = sequence_a[0:(self.max_seq_len - 3) // 2]
            if len(sequence_b) > ((self.max_seq_len - 3) // 2):
                sequence_b = sequence_b[0:(self.max_seq_len - 3) // 2]
        elif truncation_method == 'last':
            if len(sequence_b) > (self.max_seq_len - 3 - len(sequence_a)):
                sequence_b = sequence_b[0:(self.max_seq_len - 3 - len(sequence_a))]
        elif truncation_method == 'first':
            if len(sequence_a) > (self.max_seq_len - 3 - len(sequence_b)):
                sequence_a = sequence_a[0:(self.max_seq_len - 3 - len(sequence_b))]
        else:
            raise ValueError("The truncation method does not exist")

        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence_a + ['[SEP]'] + sequence_b + ['[SEP]']
        segment_ids = [0] * (len(sequence_a) + 2) + [1] * (len(sequence_b) + 1)

        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')

        return (sequence, sequence_mask, segment_ids, sequence_length) if return_sequence_length else (
            sequence, sequence_mask, segment_ids)


class SentenceTokenizer(TransfomerTokenizer):
    """
    单句分词器
    Args:
        vocab: transformers词典类对象、词典地址或词典名，用于实现文本分词和ID化
        max_seq_len (:obj:`int`): 预设的文本最大长度
    """  # noqa: ignore flake8"

    def sequence_to_ids(self, sequence_a: Union[str, List[str]], sequence_b: Union[str, List[str]] = None,
                        **kwargs) -> Tuple:
        return self.sentence_to_ids(sequence_a, **kwargs)


class PairTokenizer(TransfomerTokenizer):
    """
    句子对分词器
    Args:
        vocab: transformers词典类对象、词典地址或词典名，用于实现文本分词和ID化
        max_seq_len (:obj:`int`): 预设的文本最大长度
    """  # noqa: ignore flake8"

    def sequence_to_ids(self, sequence_a: Union[str, List[str]], sequence_b: Union[str, List[str]] = None,
                        **kwargs) -> Tuple:
        return self.pair_to_ids(sequence_a, sequence_b, **kwargs)


class TokenTokenizer(TransfomerTokenizer):
    """
    字符分词器，不
    Args:
        vocab: transformers词典类对象、词典地址或词典名，用于实现文本分词和ID化
        max_seq_len (:obj:`int`): 预设的文本最大长度
    """  # noqa: ignore flake8"

    def tokenize(self, text, **kwargs):
        tokens = []
        for token in text:
            if token == ' ':
                tokens.extend([token])
            tokens.extend(self.vocab.tokenize(token))
        return tokens


class SpanTokenizer(TransfomerTokenizer):
    """
    span 分词器
    Args:
        vocab: transformers词典类对象、词典地址或词典名，用于实现文本分词和ID化
        max_seq_len (:obj:`int`): 
            预设的文本最大长度
        split_token (:obj:`string`):
            分隔符
        additional_special_split_token (:obj:`string`):
            额外添加的特殊字符
    """  # noqa: ignore flake8"

    def __init__(self, vocab: Union[BertTokenizer, AutoTokenizer, BaseVocab], max_seq_len: int, split_token: str = ' ',
                 additional_special_split_token: str = '[unused1]'):
        super(SpanTokenizer, self).__init__(vocab, max_seq_len)
        self.split_token = split_token
        self.additional_special_split_token = additional_special_split_token if additional_special_split_token else self.split_token

        if self.additional_special_split_token not in self.vocab.vocab:
            warnings.warn(
                f"additional_special_split_token='{self.additional_special_split_token}'并不在预训练模型的词典"
                "因此,请为其在预训练模型中增加词典嵌入或更换additional_special_split_token。"
                "特别注意的是'nghuyong/ernie-1.0'预先预留了部分嵌入位置。"
            )

        self.vocab.add_special_tokens({'additional_special_tokens': [self.additional_special_split_token]})
        self.additional_special_tokens = {self.additional_special_split_token}

    def tokenize(self, text: str, **kwargs) -> List[str]:
        tokens = []
        for span_ in text.split(self.split_token):
            tokens += self.vocab.tokenize(span_)
            tokens += [self.additional_special_split_token]
        return tokens[:-1]


class PromptMLMTransformerTokenizer(TransfomerTokenizer):
    """
    模板学习Transfomer文本编码器, 用于对文本进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名, 用于实现文本分词和ID化
        max_seq_len (int): 预设的文本最大长度
    """

    def sequence_to_ids(self, sequence, prompt, prompt_mode='postfix', **kwargs):
        """
        将序列ID化
        Args:
            sequence (string or list): 输入序列
            prompt (list): 所使用的prompt, 如["是", "[MASK]"]
            prompt_mode (string):
                prompt放置在文本中的方式
                有postfix和prefix两种, postfix表示text + prompt, prefix表示prompt + text
                默认值为"postfix"
        """  # noqa: ignore flake8"

        if type(sequence) == str:
            sequence = self.tokenize(sequence)

        # 对超长序列进行截断
        if len(sequence) > self.max_seq_len - 2 - len(prompt):
            sequence = sequence[0:(self.max_seq_len - 2 - len(prompt))]

        # 分别在首尾拼接特殊符号
        if prompt_mode == 'postfix':
            sequence = ['[CLS]'] + sequence + prompt + ['[SEP]']
        else:
            sequence = ['[CLS]'] + prompt + sequence + ['[SEP]']

        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        segment_ids = [0] * len(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')

        return sequence, sequence_mask, segment_ids, len(sequence)


class ErnieCtmTokenizer(TransfomerTokenizer):
    """
    ErnieCtm模型的文本编码器，用于对文本进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名，用于实现文本分词和ID化
        max_seq_len (int): 预设的文本最大长度
        cls_num (int): CLS类特殊字符的数量，例如"[CLS0]"、"[CLS1]", 默认值为2
    """  # noqa: ignore flake8"

    def __init__(self, vocab: Union[BertTokenizer, AutoTokenizer, BaseVocab], max_seq_len: int, cls_num: int = 2):

        super().__init__(vocab, max_seq_len, max_seq_len)
        self.additional_special_tokens = set()

        self.cls_num = cls_num
        self.cls_list = ['[CLS{}]'.format(index) for index in range(self.cls_num)]

        self.do_lower_case = self.vocab.do_lower_case
        self.vocab._tokenize = self._tokenize

    def sentence_to_ids(self, sequence: Union[str, List[str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if type(sequence) == str:
            sequence = self.tokenize(sequence)

        # 对超长序列进行截断
        if len(sequence) > self.max_seq_len - 1 - self.cls_num:
            sequence = sequence[0:(self.max_seq_len - 1 - self.cls_num)]
        # 分别在首尾拼接特殊符号
        sequence = self.cls_list + sequence + ['[SEP]']
        segment_ids = [0] * len(sequence)
        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')

        return sequence, sequence_mask, segment_ids, len(sequence)

    def pair_to_ids(self, sequence_a: Union[str, List[str]], sequence_b: Union[str, List[str]],
                    truncation_method: str = 'average') -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        if type(sequence_a) == str:
            sequence_a = self.tokenize(sequence_a)
        if type(sequence_b) == str:
            sequence_b = self.tokenize(sequence_b)

        # 对超长序列进行截断
        if truncation_method == 'average':
            if len(sequence_a) > ((self.max_seq_len - 2 - self.cls_num) // 2):
                sequence_a = sequence_a[0:(self.max_seq_len - 2 - self.cls_num) // 2]
            if len(sequence_b) > ((self.max_seq_len - 2 - self.cls_num) // 2):
                sequence_b = sequence_b[0:(self.max_seq_len - 2 - self.cls_num) // 2]
        elif truncation_method == 'last':
            if len(sequence_b) > (self.max_seq_len - 2 - self.cls_num - len(sequence_a)):
                sequence_b = sequence_b[0:(self.max_seq_len - 2 - self.cls_num - len(sequence_a))]
        elif truncation_method == 'first':
            if len(sequence_a) > (self.max_seq_len - 2 - self.cls_num - len(sequence_b)):
                sequence_a = sequence_a[0:(self.max_seq_len - 2 - self.cls_num - len(sequence_b))]
        else:
            raise ValueError("The truncation method does not exist")

        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence_a + ['[SEP]'] + sequence_b + ['[SEP]']
        segment_ids = [0] * (len(sequence_a) + 1 + self.cls_num) + [1] * (len(sequence_b) + 1)

        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')

        return sequence, sequence_mask, segment_ids, (len(sequence_a), len(sequence_b))

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return [token.lower() if self.do_lower_case else token for token in list(text)]
