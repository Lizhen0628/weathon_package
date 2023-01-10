# -*- coding: utf-8 -*-
# @Time    : 2022/10/6 10:51
# @Author  : LiZhen
# @FileName: string_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

import os
import html
import pypinyin
import regex as re
import unicodedata
import urllib
import urllib.parse
import w3lib.html
from typing import List, Union
from opencc import OpenCC
from weathon.utils.char_utils import CharUtils
from weathon.utils.dictionary import Dictionary
from weathon.utils.string_converter import ConvertMap, Converter


def add_curr_dir(name):
    return os.path.join(os.path.dirname(__file__), name)


class StringUtils:

    @staticmethod
    def cut_sentences(para, drop_empty_line=True, strip=True, deduplicate=False):
        '''cut_sentences

        :param para: 输入文本
        :param drop_empty_line: 是否丢弃空行
        :param strip: 是否对每一句话做一次strip
        :param deduplicate: 是否对连续标点去重，帮助对连续标点结尾的句子分句
        :return: sentences: list of str
        '''
        if deduplicate:
            para = re.sub(r"([。！？\!\?])\1+", r"\1", para)

        if language == 'en':
            from nltk import sent_tokenize
            sents = sent_tokenize(para)
            if strip:
                sents = [x.strip() for x in sents]
            if drop_empty_line:
                sents = [x for x in sents if len(x.strip()) > 0]
            return sents
        else:
            para = re.sub('([。！？\?!])([^”’)\]）】])', r"\1\n\2", para)  # 单字符断句符
            para = re.sub('(\.{3,})([^”’)\]）】….])', r"\1\n\2", para)  # 英文省略号
            para = re.sub('(\…+)([^”’)\]）】….])', r"\1\n\2", para)  # 中文省略号
            para = re.sub('([。！？\?!]|\.{3,}|\…+)([”’)\]）】])([^，。！？\?….])', r'\1\2\n\3', para)
            # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
            para = para.rstrip()  # 段尾如果有多余的\n就去掉它
            # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
            sentences = para.split("\n")
            if strip:
                sentences = [sent.strip() for sent in sentences]
            if drop_empty_line:
                sentences = [sent for sent in sentences if len(sent.strip()) > 0]
            return sentences

    @staticmethod
    def clean_text(text, remove_url=False, email=False, weibo_at=False, stop_terms=("转发微博",),
                   emoji=False, weibo_topic=False, deduplicate_space=True,
                   norm_url=False, norm_html=False, to_url=False,
                   remove_puncts=False, remove_tags=True, t2s=True, q2b=True, wx_emoji2word=True,
                   expression_len=(1, 6), linesep2space=False):

        """
        进行各种文本清洗操作，微博中的特殊格式，网址，email，html代码，等等
        Args:
            text:输入文本
            remove_url:（默认不使用）是否去除网址
            email:（默认不使用）是否去除email
            weibo_at:（默认不使用）是否去除微博的\@相关文本
            stop_terms: 去除文本中的一些特定词语，默认参数为("转发微博",)
            emoji: （默认不使用）去除\[\]包围的文本，一般是表情符号
            weibo_topic:（默认不使用）去除##包围的文本，一般是微博话题
            deduplicate_space:（默认使用）合并文本中间的多个空格为一个
            norm_url:（默认不使用）还原URL中的特殊字符为普通格式，如(%20转为空格)
            norm_html: （默认不使用）还原HTML中的特殊字符为普通格式，如(\&nbsp;转为空格)
            to_url:  （默认不使用）将普通格式的字符转为还原URL中的特殊字符，用于请求，如(空格转为%20)
            remove_puncts: （默认不使用）移除所有标点符号
            remove_tags: （默认使用）移除所有html块
            t2s: （默认使用）繁体字转中文
            q2b: (默认使用) 全角转半角
            expression_len: 假设表情的表情长度范围，不在范围内的文本认为不是表情，不加以清洗，如[加上特别番外荞麦花开时共五册]。设置为None则没有限制
            linesep2space: （默认不使用）把换行符转换成空格
        Returns: 清洗后的文本
        """

        text = StringUtils.as_text(text)
        # unicode不可见字符
        # 未转义
        text = re.sub(r"[\u200b-\u200d]", "", text)
        # 已转义
        text = re.sub(r"(\\u200b|\\u200c|\\u200d)", "", text)
        # 反向的矛盾设置
        if norm_url and to_url:
            raise Exception("norm_url和to_url是矛盾的设置")
        if norm_html:
            text = html.unescape(text)
        if to_url:
            text = urllib.parse.quote(text)
        if remove_tags:
            text = w3lib.html.remove_tags(text)
        if remove_url:
            try:
                URL_REGEX = re.compile(
                    r'(?i)http[s]?://(?:[a-zA-Z]|[0-9]|[#$%*-;=?&@~.&+]|[!*,])+',
                    re.IGNORECASE)
                text = re.sub(URL_REGEX, "", text)
            except:
                # sometimes lead to "catastrophic backtracking"
                zh_puncts1 = "，；、。！？（）《》【】"
                URL_REGEX = re.compile(
                    r'(?i)((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>' + zh_puncts1 + ']+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’' + zh_puncts1 + ']))',
                    re.IGNORECASE)
                text = re.sub(URL_REGEX, "", text)
        if norm_url:
            text = urllib.parse.unquote(text)
        if email:
            EMAIL_REGEX = re.compile(r"[-a-z0-9_.]+@(?:[-a-z0-9]+\.)+[a-z]{2,6}", re.IGNORECASE)
            text = re.sub(EMAIL_REGEX, "", text)
        if weibo_at:
            text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:|：| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
        if wx_emoji2word:
            text = StringUtils.wechat_expression_2_word(text)
        if emoji:
            # 去除括号包围的表情符号
            # ? lazy match避免把两个表情中间的部分去除掉
            if type(expression_len) in {tuple, list} and len(expression_len) == 2:
                # 设置长度范围避免误伤人用的中括号内容，如[加上特别番外荞麦花开时共五册]
                lb, rb = expression_len
                text = re.sub(r"\[\S{" + str(lb) + r"," + str(rb) + r"}?\]", "", text)
            else:
                text = re.sub(r"\[\S+?\]", "", text)
            # text = re.sub(r"\[\S+\]", "", text)
            # 去除真,图标式emoji
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       u"\U00002702-\U000027B0"
                                       "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
        if weibo_topic:
            text = re.sub(r"#\S+#", "", text)  # 去除话题内容
        if linesep2space:
            text = text.replace("\n", " ")  # 不需要换行的时候变成1行
        if deduplicate_space:
            text = re.sub(r"(\s)+", r"\1", text)  # 合并正文中过多的空格
        if t2s:
            text = StringUtils.traditional2simple(text)
        assert hasattr(stop_terms, "__iter__"), Exception("去除的词语必须是一个可迭代对象")
        if type(stop_terms) == str:
            text = text.replace(stop_terms, "")
        else:
            for x in stop_terms:
                text = text.replace(x, "")
        if remove_puncts:
            text = StringUtils.remove_string_punc(text)
        if q2b:
            text = StringUtils.Q2B(text)
        return text.strip()

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
    def is_control(ch: str) -> bool:
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def is_special(ch: str) -> bool:
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

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
    def traditional2japanese(text: str) -> str:
        """
        日文新字体 转化 为繁体
        """
        return OpenCC("jp2t").convert(text)

    @staticmethod
    def Japanese2traditional(text: str) -> str:
        """
        繁體（OpenCC 標準，舊字體）到日文新字體  繁体转化为日文新字体
        """
        return OpenCC("t2jp").convert(text)

    @staticmethod
    def traditional2traditional(text: str, convert_type=""):
        """
        Args:
            text: 原始字符串
            convert_type: 转化类型
                t2tw.json Traditional Chinese (OpenCC Standard) to Taiwan Standard 繁體（OpenCC 標準）到臺灣正體
                hk2t.json Traditional Chinese (Hong Kong variant) to Traditional Chinese 香港繁體到繁體（OpenCC 標準）
                t2hk.json Traditional Chinese (OpenCC Standard) to Hong Kong variant 繁體（OpenCC 標準）到香港繁體
                tw2t.json Traditional Chinese (Taiwan standard) to Traditional Chinese 臺灣正體到繁體（OpenCC 標準）
        Returns:
        """
        assert len(convert_type) != 0, "convert_type is none, must be in ['t2tw','hk2t','t2hk','tw2t']"
        return OpenCC(convert_type).convert(text)

    @staticmethod
    def traditional2simple(text: str, convert_type="t2s") -> str:
        """
        将繁体字符串转化为简体字符串
        Args:
            text:
            convert_type:
                t2s.json Traditional Chinese to Simplified Chinese 繁體到簡體
                tw2s.json Traditional Chinese (Taiwan Standard) to Simplified Chinese 臺灣正體到簡體
                hk2s.json Traditional Chinese (Hong Kong variant) to Simplified Chinese 香港繁體到簡體
                tw2sp.json Traditional Chinese (Taiwan Standard) to Simplified Chinese with Mainland Chinese idiom 繁體（臺灣正體標準）到簡體並轉換爲中國大陸常用詞彙
        """
        # word_map = Dictionary.traditional2simple_dic()
        # return Converter(ConvertMap(word_map)).convert(text)
        return OpenCC(convert_type).convert(text)

    @staticmethod
    def simple2traditional(text: str, convert_type="s2t") -> str:
        """
        将简体字符串转化为繁体字符串
        Args:
            text:
            convert_type:
                s2t.json Simplified Chinese to Traditional Chinese 简体到繁體
                s2tw.json Simplified Chinese to Traditional Chinese (Taiwan Standard) 簡體到臺灣正體
                s2hk.json Simplified Chinese to Traditional Chinese (Hong Kong variant) 簡體到香港繁體
                s2twp.json Simplified Chinese to Traditional Chinese (Taiwan Standard) with Taiwanese idiom 簡體到繁體（臺灣正體標準）並轉換爲臺灣常用詞彙
        """
        # word_map = Dictionary.simple2traditional_dic()
        # return Converter(ConvertMap(word_map)).convert(text)
        return OpenCC(convert_type).convert(text)

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
    def remove_string_punc(s: str) -> str:
        """去除字符串中的标点符号"""
        s = re.compile("[。，！？、；：“”‘’（）【】\{\}『』「」〔〕——……—\-～·《》〈〉﹏___\.]").sub("", s)
        s = re.compile(
            "[,\.\":\)\(\-!\?\|;'\$&/\[\]>%=#\*\+\\•~@£·_\{\}©\^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√]").sub(
            "", s)
        return s


if __name__ == '__main__':
    # print(StringUtils.simple2traditional(
    #     "现代社会，很多人都想有个红颜知己，也就是异性朋友，在自己孤独寂寞时，可以有个出口。只是异性朋友在一起，难免会引人遐想，或许你只是单纯地认为，有个这样的朋友，能让自己可以放心诉说心事，还能够保持单纯的朋友关系。"))
    # --------------- 测试文本清洗 ------------------


    # 不可见字符
    # text1 = "捧杀！干得漂亮！[doge] \\u200b\\u200b\\u200b"
    # text2 = StringUtils.clean_text(text1)
    # print("清洗前：", [text1])
    # print("清洗后：", [text2])
    # assert text2 == "捧杀！干得漂亮！"

    # 两个表情符号中间有内容
    # text1 = "#缺钱找新浪# 瞎找不良网贷不如用新浪官方借款，不查负债不填联系人。  http://t.cn/A643boyi \n新浪[浪]用户专享福利，[浪]新浪产品用的越久额度越高，借万元日利率最低至0.03%，最长可分12期慢慢还！ http://t.cn/A643bojv  http://t.cn/A643bKHS \u200b\u200b\u200b"
    # text2 = StringUtils.clean_text(text1,remove_url=True,weibo_topic=True)
    # print("清洗前：", [text1])
    # print("清洗后：", [text2])

    # 包含emoji
    text1 = "各位大神们🙏求教一下这是什么动物呀！[疑问]\n\n为什么它同时长得有点吓人又有点可爱[允悲]\n\n#thosetiktoks# http://t.cn/A6bXIC44 \u200b\u200b\u200b"
    text2 = StringUtils.clean_text(text1,emoji=True,remove_url=True)
    print("清洗前：", [text1])
    print("清洗后：", [text2])
    # text1 = "JJ棋牌数据4.3万。数据链接http://www.jj.cn/，数据第一个账号，第二个密码，95%可登录，可以登录官网查看数据是否准确"
    # text2 = StringUtils.clean_text(text1)
    # assert text2 == "JJ棋牌数据4.3万。数据链接，数据第一个账号，第二个密码，95%可登录，可以登录官网查看数据是否准确"