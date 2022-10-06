# -*- coding: utf-8 -*-
# @Time    : 2022/10/6 10:51
# @Author  : LiZhen
# @FileName: string_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

import os
import pypinyin
import regex as re
from pathlib import Path
from typing import List, Union
from collections import defaultdict
from weathon.utils.char_utils import CharUtils
from weathon.utils.dictionary import Dictionary
from weathon.utils.string_converter import ConvertMap, Converter

CNPuncs_rule = re.compile("[。，！？、；：“”‘’（）【】\{\}『』「」〔〕——……—\-～·《》〈〉﹏___\.]")

ENPuncs_rule = re.compile(
    "[,\.\":\)\(\-!\?\|;'\$&/\[\]>%=#\*\+\\•~@£·_\{\}©\^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√]")


def add_curr_dir(name):
    return os.path.join(os.path.dirname(__file__), name)


class StringUtils:

    @staticmethod
    def cut_sentences(sentence: str) -> List[str]:
        sentence_delimiters = ['。', '？', '！', '…']
        sents = []
        tmp = []
        for ch in sentence:  # 遍历字符串中的每一个字
            tmp.append(ch)
            if ch in sentence_delimiters:
                sents.append(''.join(tmp))
                tmp = []
        if len(tmp) > 0:  # 如以定界符结尾的文本的文本信息会在循环中返回，无需再次传递
            sents.append(''.join(tmp))
        return sents

    @staticmethod
    def is_number_string(string: str) -> bool:
        """判断字符串是否全为数字"""
        return all(CharUtils.is_number(c) for c in string)

    @staticmethod
    def is_chinese_string(string: str) -> bool:
        """判断字符串是否全为汉字"""
        return all(CharUtils.is_chinese(c) for c in string)

    @staticmethod
    def is_eng_string(string: str) -> bool:
        """判断字符串是否全为英文"""
        return all(CharUtils.is_alphabet(c) for c in string)

    @staticmethod
    def escape_re_character(s: str) -> str:
        """对正则字符添加转义"""
        escape_char = ["\\", "\"", "*", ".", "?", "+", "$", "^", "[", "]", "(", ")", "{", "}", "|", "-"]
        for c in escape_char:
            if c in s:
                s = s.replace(c, re.escape(c))
        return s

    @staticmethod
    def B2Q(text: str) -> str:
        """
        字符串半角转全角
        Args:
            text: 输入字符串

        Returns:转化后字符串
        """
        return "".join([CharUtils.B2Q(c) for c in text])

    @staticmethod
    def Q2B(text: str) -> str:
        """
        字符串全角转半角
        Args:
            text: 输入字符串

        Returns:转化后字符串
        """
        return "".join([CharUtils.Q2B(c) for c in text])

    @staticmethod
    def as_text(text: str) -> Union[str, None]:
        """生成unicode字符串"""
        if text is None:
            return None
        elif isinstance(text, bytes):
            try:
                text = text.decode('utf-8')
            except UnicodeDecodeError:
                text = text.decode('gbk', 'ignore')
            return text
        elif isinstance(text, str):
            return text
        else:
            raise ValueError('Unknown type %r' % type(text))

    @staticmethod
    def wechat_expression_2_word(text: str) -> str:
        """将文本中的微信表情代码转化成文字"""
        code_expression = Dictionary.wechat_expression()
        for key, value in code_expression.items():
            text = text.replace(key, value)
        return text

    @staticmethod
    def traditional2simple(text: str) -> str:
        """
        将繁体字符串转化为简体字符串
        """
        word_map = Dictionary.traditional2simple_dic()
        return Converter(ConvertMap(word_map)).convert(text)

    @staticmethod
    def simple2traditional(text: str) -> str:
        """
        将简体字符串转化为繁体字符串
        """
        word_map = Dictionary.simple2traditional_dic()
        return Converter(ConvertMap(word_map)).convert(text)

    @staticmethod
    def text2pinyin(text: str, with_tone: bool = False) -> List[str]:
        """
        将文本字符串转成拼音
        Args:
            text: 输入的文本字符串
            with_tone: 是否对转化的拼音加上声调
        Returns:
        """
        result = pypinyin.core.pinyin(text, strict=False, style=pypinyin.TONE if with_tone else pypinyin.NORMAL)
        return [_[0] for _ in result]

    @staticmethod
    def get_homophones_by_pinyin(input_pinyin: str) -> List[str]:
        """根据拼音取所有同音字"""
        result = []
        # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
        for i in range(0x4e00, 0x9fa6):
            if pypinyin.core.pinyin([chr(i)], style=pypinyin.NORMAL, strict=False)[0][0] == input_pinyin:
                result.append(chr(i))
        return result

    @staticmethod
    def remove_string_punc(cls, s: str) -> str:
        """去除字符串中的标点符号"""
        s = CNPuncs_rule.sub("", s)
        s = ENPuncs_rule.sub("", s)
        return s


if __name__ == '__main__':
    print(StringUtils.simple2traditional(
        "现代社会，很多人都想有个红颜知己，也就是异性朋友，在自己孤独寂寞时，可以有个出口。只是异性朋友在一起，难免会引人遐想，或许你只是单纯地认为，有个这样的朋友，能让自己可以放心诉说心事，还能够保持单纯的朋友关系。"))
