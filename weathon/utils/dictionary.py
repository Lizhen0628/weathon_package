# -*- coding: utf-8 -*-
# @Time    : 2022/10/6 15:25
# @Author  : LiZhen
# @FileName: dictionary.py
# @github  : https://github.com/Lizhen0628
# @Description:
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set


def make_dic(lines: List[str]) -> Dict:
    word_map = defaultdict()
    for line in lines:
        splits = line.split()
        if len(splits) == 2:
            word_map[splits[0]] = splits[1]

    return word_map


class Dictionary:

    @staticmethod
    def simple2traditional_dic() -> Dict:
        file = Path(__file__).parent.parent / "data/dictionary/simple2traditional.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def traditional2simple_dic():
        file = Path(__file__).parent.parent / "data/dictionary/traditional2simple.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def wechat_expression():
        file = Path(__file__).parent.parent / "data/dictionary/wechat_expression.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def place_name():
        """
        地名词库
        place_name:freq
        {
            "医院":"562296",
            "学校":"523819",
            "法院":"367721"
        }
        """
        file = Path(__file__).parent.parent / "data/dictionary/place_name.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def medical_name() -> Dict:
        """
        医疗领域词库
        """
        file = Path(__file__).parent.parent / "data/dictionary/medical.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def antonym_word() -> Dict:
        """
        反义词库
        """
        file = Path(__file__).parent.parent / "data/dictionary/antonym_word.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def car() -> Dict:
        """
        汽车领域词库
        """
        file = Path(__file__).parent.parent / "data/dictionary/car.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def finance() -> Dict:
        """
        金融领域词库
        """
        file = Path(__file__).parent.parent / "data/dictionary/finance.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def legal() -> Dict:
        """
        法律领域词库
        """
        file = Path(__file__).parent.parent / "data/dictionary/legal.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def animal() -> Dict:
        """
        动物名词库
        """
        file = Path(__file__).parent.parent / "data/dictionary/animal.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def food() -> Dict:
        """
        食物词 词库，包含8k多个食物词汇，格式：土豆	1777511
        """
        file = Path(__file__).parent.parent / "data/dictionary/food.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def family_name() -> Dict:
        """
        姓氏词表 format: 姓 词频 ,刘 5086
        """
        file = Path(__file__).parent.parent / "data/dictionary/family_name.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def professions() -> Dict:
        """
        职业名称词库
        """
        file = Path(__file__).parent.parent / "data/dictionary/professions.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def it() -> Dict:
        """
        it 词库，包含16k个互联网计算机相关词汇，格式：数组 	 357924
        """
        file = Path(__file__).parent.parent / "data/dictionary/it.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def same_stroke() -> Dict:
        """
        形近字 词库
        """
        file = Path(__file__).parent.parent / "data/dictionary/same_stroke.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def same_pinyin() -> Dict:
        """
        音近字 词库
        """
        file = Path(__file__).parent.parent / "data/dictionary/same_pinyin.txt"
        return make_dic(file.read_text().split("\n"))

    @staticmethod
    def stopwords() -> Set:
        """
        音近字 词库
        """
        file = Path(__file__).parent.parent / "data/dictionary/stopwords.txt"
        return {line.strip() for line in file.read_text().split("\n")}


