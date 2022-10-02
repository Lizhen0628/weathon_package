# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 16:51
# @Author  : LiZhen
# @FileName: token_classification_dataset.py.py
# @github  : https://github.com/Lizhen0628
# @Description:

import torch
import numpy as np
from weathon.nlp.base import BaseDataset


class TokenClassificationDataset(BaseDataset):
    """
    用于字符分类任务的Dataset
    """

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text'] = data_df['text'].apply(lambda x: x.strip())
        if not self.is_test:
            data_df['label'] = data_df['label'].apply(
                lambda x: eval(x) if type(x) == str else x)

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        return dataset


class SpanNERDataset(TokenClassificationDataset):
    """
    用于Span模式的命名实体识别任务的Dataset
    """  # noqa: ignore flake8"

    def _get_categories(self):
        categories = sorted(list(set([label_['type']
                                      for data in self.dataset for label_ in data['label']])))
        if 'O' in categories:
            categories.remove('O')
        categories.insert(0, 'O')
        return categories

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            tokens = bert_tokenizer.tokenize(row_['text'])[:bert_tokenizer.max_seq_len - 2]
            token_mapping = bert_tokenizer.get_token_mapping(row_['text'], tokens)

            start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}

            input_ids = bert_tokenizer.sequence_to_ids(tokens)

            input_ids, input_mask, segment_ids = input_ids

            start_label = torch.zeros((bert_tokenizer.max_seq_len))

            end_label = torch.zeros((bert_tokenizer.max_seq_len))

            label_ = set()
            for info_ in row_['label']:
                if info_['start_idx'] in start_mapping and info_['end_idx'] in end_mapping:
                    start_idx = start_mapping[info_['start_idx']]
                    end_idx = end_mapping[info_['end_idx']]
                    if start_idx > end_idx or info_['entity'] == '':
                        continue

                    start_label[start_idx + 1] = self.cat2id[info_['type']]
                    end_label[end_idx + 1] = self.cat2id[info_['type']]

                    label_.add((self.cat2id[info_['type']], start_idx, end_idx))

            features.append({
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'start_label_ids': start_label,
                'end_label_ids': end_label,
                'label_ids': list(label_)
            })

        return features

    @property
    def to_device_cols(self):
        _cols = list(self.dataset[0].keys())
        _cols.remove('label_ids')
        return _cols


class GlobalPointerNERDataset(TokenClassificationDataset):
    """
    用于GlobalPointer命名实体识别任务的Dataset
    """

    def _get_categories(self):
        categories = sorted(list(set([label_['type'] for data in self.dataset for label_ in data['label']])))
        return categories

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            tokens = bert_tokenizer.tokenize(row_['text'])[:bert_tokenizer.max_seq_len - 2]
            token_mapping = bert_tokenizer.get_token_mapping(row_['text'], tokens)

            start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}

            input_ids = bert_tokenizer.sequence_to_ids(tokens)

            input_ids, input_mask, segment_ids = input_ids

            global_label = torch.zeros((
                self.class_num,
                bert_tokenizer.max_seq_len,
                bert_tokenizer.max_seq_len)
            )

            for info_ in row_['label']:
                if info_['start_idx'] in start_mapping and info_['end_idx'] in end_mapping:
                    start_idx = start_mapping[info_['start_idx']]
                    end_idx = end_mapping[info_['end_idx']]
                    if start_idx > end_idx or info_['entity'] == '':
                        continue
                    global_label[self.cat2id[info_['type']], start_idx + 1, end_idx + 1] = 1

            global_label = global_label.to_sparse()

            features.append({
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'label_ids': global_label
            })

        return features


class BIONERDataset(TokenClassificationDataset):
    """
    用于BIO形式的字符分类任务的Dataset
    """

    def _get_categories(self):

        categories = []
        types_ = set([label_['type'] for data in self.dataset for label_ in data['label']])
        for type_ in types_:
            categories.append('B-' + type_)
            categories.append('I-' + type_)

        categories = sorted(categories)

        if 'O' in categories:
            categories.remove('O')
        categories.insert(0, 'O')
        return categories

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            tokens = bert_tokenizer.tokenize(row_['text'])[:bert_tokenizer.max_seq_len - 2]
            token_mapping = bert_tokenizer.get_token_mapping(row_['text'], tokens)

            start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}

            input_ids = bert_tokenizer.sequence_to_ids(tokens)

            input_ids, input_mask, segment_ids = input_ids
            input_length = len(tokens)

            feature = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'input_lengths': input_length
            }

            if not self.is_test:
                label_ids = len(input_ids) * [self.cat2id['O']]

                for info_ in row_['label']:
                    if info_['start_idx'] in start_mapping and info_['end_idx'] in end_mapping:
                        start_idx = start_mapping[info_['start_idx']]
                        end_idx = end_mapping[info_['end_idx']]
                        if start_idx > end_idx or info_['entity'] == '':
                            continue

                        label_ids[start_idx + 1] = self.cat2id['B-' + info_['type']]

                        label_ids[start_idx + 2:end_idx + 2] = [self.cat2id['I-' + info_['type']]] * (
                                    end_idx - start_idx)
                feature['label_ids'] = np.array(label_ids)

            features.append(feature)

        return features

class BiaffineNERDataset(TokenClassificationDataset):
    """
    用于Biaffine命名实体识别任务的Dataset
    """

    def _get_categories(self):
        categories = sorted(list(set([label_['type'] for data in self.dataset for label_ in data['label']])))
        if 'O' in categories:
            categories.remove('O')
        categories.insert(0, 'O')
        return categories

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            tokens = bert_tokenizer.tokenize(row_['text'])[:bert_tokenizer.max_seq_len-2]
            token_mapping = bert_tokenizer.get_token_mapping(row_['text'], tokens)

            start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}

            input_ids = bert_tokenizer.sequence_to_ids(tokens)

            input_ids, input_mask, segment_ids = input_ids

            zero = [0 for i in range(bert_tokenizer.max_seq_len)]
            span_mask = [input_mask for _ in range(sum(input_mask))]
            span_mask.extend([zero for _ in range(sum(input_mask),
                                                  bert_tokenizer.max_seq_len)])
            span_mask = np.array(span_mask)

            span_label = [0 for _ in range(bert_tokenizer.max_seq_len)]
            span_label = [span_label for _ in range(bert_tokenizer.max_seq_len)]
            span_label = np.array(span_label)

            for info_ in row_['label']:
                if info_['start_idx'] in start_mapping and info_['end_idx'] in end_mapping:
                    start_idx = start_mapping[info_['start_idx']]
                    end_idx = end_mapping[info_['end_idx']]
                    if start_idx > end_idx or info_['entity'] == '':
                        continue

                    span_label[start_idx+1, end_idx+1] = self.cat2id[info_['type']]

            features.append({
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'label_ids': span_label,
                'span_mask': span_mask
            })

        return features
