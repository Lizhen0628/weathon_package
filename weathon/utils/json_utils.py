# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 17:09
# @Author  : LiZhen
# @FileName: json_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:


from typing import Dict


class JsonUtils:

    @staticmethod
    def key_to_int(dic: Dict) -> Dict:
        """
        加载json文件时，将 字符串类型 的key转化成 整数类型 的key
        Args:
            dic: 原字典

        Returns:转化后的字典类型
        如果key中不全为数字，返回原字典
        """
        try:
            return {int(k): v for k, v in dic.items()}
        except ValueError:
            return dic
