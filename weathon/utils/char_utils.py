# -*- coding: utf-8 -*-
# @Time    : 2022/10/6 10:26
# @Author  : LiZhen
# @FileName: char_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:
from typing import List


class CharUtils:

    @staticmethod
    def B2Q(uchar: str) -> str:
        """单字符半角转全角"""
        assert len(uchar) == 1, "uchar 只能是单个字符"
        inside_code = ord(uchar)
        if inside_code < 0x0020 or inside_code > 0x7e:
            # 不是半角字符就返回原来的字符
            return uchar
        if inside_code == 0x0020:
            # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
            inside_code = 0x3000
        else:
            inside_code += 0xfee0
        return chr(inside_code)

    @staticmethod
    def Q2B(uchar: str) -> str:
        """单字符全角转半角"""
        assert len(uchar) == 1, "uchar 只能是单个字符"
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:
            # 转完之后不是半角字符返回原来的字符
            return uchar
        return chr(inside_code)


    @staticmethod
    def is_number(uchar):
        """判断一个字符是否是数字"""
        assert len(uchar) == 1, "uchar 只能是单个字符"
        return '\u0030' <= uchar <= '\u0039'


    @staticmethod
    def is_alphabet(uchar):
        """判断一个字符是否是英文字母"""
        assert len(uchar) == 1, "uchar 只能是单个字符"
        return (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a')


    @staticmethod
    def is_chinese(ch:str) -> bool:
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        cp = ord(ch)
        return ((0x4E00 <= cp <= 0x9FFF) or  #
                (0x3400 <= cp <= 0x4DBF) or  #
                (0x20000 <= cp <= 0x2A6DF) or  #
                (0x2A700 <= cp <= 0x2B73F) or  #
                (0x2B740 <= cp <= 0x2B81F) or  #
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or  #
                (0x2F800 <= cp <= 0x2FA1F))  #

