# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 00:15
# @Author  : LiZhen
# @FileName: encrypt_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

from hashlib import md5


class EncryptUtils:
    """
    加密工具类
    """

    @staticmethod
    def encry_md5(in_str: str, salt: str = "123456") -> str:
        """
        输入 字符串，对字符串采用md5算法加密，
        :param in_str:  需要加密的字符串
        :param salt:  盐，防止被撞库
        :return:  对字符串加密后的结果
        """

        obj = md5(salt.encode("utf8"))
        obj.update(in_str.encode("utf8"))
        return obj.hexdigest()
