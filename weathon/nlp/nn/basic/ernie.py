# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 16:52
# @Author  : LiZhen
# @FileName: ernie.py
# @github  : https://github.com/Lizhen0628
# @Description:

from weathon.nlp.nn.basic.bert import Bert,BertForSequenceClassification,BertForTokenClassification


class Ernie(Bert):
    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(Ernie, self).__init__(config, encoder_trained)


class ErnieForSequenceClassification(BertForSequenceClassification):
    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(ErnieForSequenceClassification, self).__init__(config, encoder_trained)


class ErnieForTokenClassification(BertForTokenClassification):
    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(BertForTokenClassification, self).__init__(config, encoder_trained)