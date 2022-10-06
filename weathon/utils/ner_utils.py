# -*- coding: utf-8 -*-
# @Time    : 2022/10/4 08:48
# @Author  : LiZhen
# @FileName: ner_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:


import json
import numpy as np
from pathlib import Path
from seqeval.metrics import sequence_labeling
from typing import Union


class NERUtils:

    @staticmethod
    def span_extract_entities(start_pred, end_pred, text):
        """
        用于抽取 span model 所对应的实体
        :param start_pred:
        :param end_pred:
        :param text:
        :return:
        """

        S = []
        for i, s_l in enumerate(start_pred):
            if s_l == 0:
                continue
            for j, e_l in enumerate(end_pred[i:]):
                if s_l == e_l:
                    S.append((s_l, i, i + j + 1, text[i:i + j + 1]))
                    break
        return S

    @staticmethod
    def bioes_tag_to_spans(tags, ignore_labels=None):
        """
        给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'E-singer', 'O', 'O']。
        返回[('singer', (1, 4))] (左闭右开区间)
        :param tags: List[str],
        :param ignore_labels: List[str], 在该list中的label将被忽略
        :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
        """
        ignore_labels = set(ignore_labels) if ignore_labels else set()

        spans = []
        prev_bioes_tag = None
        for idx, tag in enumerate(tags):
            tag = tag.lower()
            bioes_tag, label = tag[:1], tag[2:]
            if bioes_tag in ('b', 's'):
                spans.append((label, [idx, idx]))
            elif bioes_tag in ('i', 'e') and prev_bioes_tag in ('b', 'i') and label == spans[-1][0]:
                spans[-1][1][1] = idx
            elif bioes_tag == 'o':
                pass
            else:
                spans.append((label, [idx, idx]))
            prev_bioes_tag = bioes_tag
        return [(span[0], span[1][0], span[1][1])
                for span in spans
                if span[0] not in ignore_labels
                ]

    @staticmethod
    def bmeso_tag_to_spans(tags, ignore_labels=None):
        """
        给定一个tags的lis，比如['O', 'B-singer', 'M-singer', 'E-singer', 'O', 'O']。
        返回[('singer', (1, 4))] (左闭右开区间)
        :param tags: List[str],
        :param ignore_labels: List[str], 在该list中的label将被忽略
        :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
        """
        ignore_labels = set(ignore_labels) if ignore_labels else set()

        spans = []
        prev_bmes_tag = None
        for idx, tag in enumerate(tags):
            tag = tag.lower()
            bmes_tag, label = tag[:1], tag[2:]
            if bmes_tag in ('b', 's'):
                spans.append((label, [idx, idx]))
            elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
                spans[-1][1][1] = idx
            elif bmes_tag == 'o':
                pass
            else:
                spans.append((label, [idx, idx]))
            prev_bmes_tag = bmes_tag
        return [(span[0], span[1][0], span[1][1]) for span in spans if span[0] not in ignore_labels]

    @staticmethod
    def bmes_tag_to_spans(tags, ignore_labels=None):
        """
        给定一个tags的lis，比如['S-song', 'B-singer', 'M-singer', 'E-singer', 'S-moive', 'S-actor']。
        返回[('song', (0, 1)), ('singer', (1, 4)), ('moive', (4, 5)), ('actor', (5, 6))] (左闭右开区间)
        也可以是单纯的['S', 'B', 'M', 'E', 'B', 'M', 'M',...]序列
        :param tags: List[str],
        :param ignore_labels: List[str], 在该list中的label将被忽略
        :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
        """
        ignore_labels = set(ignore_labels) if ignore_labels else set()

        spans = []
        prev_bmes_tag = None
        for idx, tag in enumerate(tags):
            tag = tag.lower()
            bmes_tag, label = tag[:1], tag[2:]
            if bmes_tag in ('b', 's'):
                spans.append((label, [idx, idx]))
            elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
                spans[-1][1][1] = idx
            else:
                spans.append((label, [idx, idx]))
            prev_bmes_tag = bmes_tag
        return [(span[0], span[1][0], span[1][1])
                for span in spans
                if span[0] not in ignore_labels
                ]

    @staticmethod
    def bio_tag_to_spans(tags, ignore_labels=None):
        """
        给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O']。
            返回[('singer', (1, 4))] (左闭右开区间)
        :param tags: List[str],
        :param ignore_labels: List[str], 在该list中的label将被忽略
        :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
        """
        ignore_labels = set(ignore_labels) if ignore_labels else set()

        spans = []
        prev_bio_tag = None
        for idx, tag in enumerate(tags):
            tag = tag.lower()
            bio_tag, label = tag[:1], tag[2:]
            if bio_tag == 'b':
                spans.append((label, [idx, idx]))
            elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
                spans[-1][1][1] = idx
            elif bio_tag == 'o':  # o tag does not count
                pass
            else:
                spans.append((label, [idx, idx]))
            prev_bio_tag = bio_tag
        return [(span[0], span[1][0], span[1][1]) for span in spans if span[0] not in ignore_labels]

    @staticmethod
    def conll2jsonl(infile: Union[str, Path], outfile: Union[str, Path]) -> None:
        infile = Path(infile)
        outfile = Path(outfile)
        words = []
        labels = []
        idx = 0

        with infile.open("r", encoding="utf8") as reader, outfile.open("w", encoding="utf8") as writer:
            for line in reader:
                if len(line) < 3:
                    text = "".join(words)
                    entities = sequence_labeling.get_entities(labels)
                    writer.write(json.dumps({"id": idx, "text": text,
                                             "labels": [[x[1], x[2] + 1, x[0], text[x[1]:(x[2] + 1)]] for x in entities]
                                             }, ensure_ascii=False) + "\n")

                    idx += 1
                    words = []
                    labels = []

                else:
                    words.append(line[0])
                    labels.append(line[2:].rstrip("\n").upper())

        print('conll --> jsonl : convert done !')

    @staticmethod
    def BIO2BMES(infile: Union[str, Path], outfile: Union[str, Path]) -> None:
        infile = Path(infile)
        outfile = Path(outfile)

        words = []
        labels = []
        with infile.open("r", encoding="utf8") as reader, outfile.open("w", encoding="utf8") as writer:
            for line in reader:
                if len(line) < 3:
                    sentence_len = len(words)
                    for idx in range(sentence_len):
                        if "-" not in labels[idx]:
                            writer.write(words[idx] + " " + labels[idx] + "\n")
                        else:
                            label_type = labels[idx].split("-")[-1]

                            if "B-" in labels[idx]:
                                if (idx == sentence_len - 1) or ("I-" not in labels[idx + 1]):
                                    writer.write(words[idx] + " S-" + label_type + "\n")
                                else:
                                    writer.write(words[idx] + " B-" + label_type + "\n")
                            elif "I-" in labels[idx]:
                                if (idx == sentence_len - 1) or ("I-" not in labels[idx + 1]):
                                    writer.write(words[idx] + " E-" + label_type + "\n")
                                else:
                                    writer.write(words[idx] + " M-" + label_type + "\n")
                    writer.write("\n")
                    words = []
                    labels = []
                else:
                    if line == "0\t\n":
                        words.append('0')
                        labels.append("O")
                    else:
                        pair = line.rstrip("\n").split()
                        words.append(pair[0])
                        labels.append(pair[-1].upper())

        print("BOI --> BMES : convert done !")

    @staticmethod
    def text2tags(text: str, labels: list, label2id: dict, label_mode: str, max_length):
        """
        将文本text中的内容，根据labels以及label_mode:["bio","bios","bieos"] 转换成 标注格式
        :param text:
        :param labels:
        :param label2id:
        :param label_mode:
        :return:
        """
        if label_mode.lower() == 'bio':
            return NERUtils.text2bio(text, labels, label2id, max_length)
        elif label_mode.lower() == 'bios':
            return NERUtils.text2bios(text, labels, label2id, max_length)
        elif label_mode.lower() == 'bioes':
            return NERUtils.text2bioes(text, labels, label2id, max_length)
        else:
            raise Exception("{} not in ['bio','bios','bioes']")
    @staticmethod
    def text2bio(text: str, labels: list, label2id: dict, max_length: int):
        label_tag = ['O'] * len(text)
        label_ids = [0] * len(text)

        for label_item in labels:
            assert text[int(label_item[0]):int(label_item[1])] == label_item[
                3], 'entity text : {} obtained by text[start:end] is not the same as label entity text:{} '.format(
                text[int(label_item[0]):int(label_item[1])], label_item[3])
            if 'B-' + label_item[2] not in label2id:
                label2id['B-' + label_item[2]] = len(label2id)
            if 'I-' + label_item[2] not in label2id:
                label2id['I-' + label_item[2]] = len(label2id)

            label_tag[int(label_item[0])] = 'B-' + label_item[2]
            label_ids[int(label_item[0])] = label2id['B-' + label_item[2]]
            if len(label_item[3]) >= 2:
                label_tag[int(label_item[0]) + 1:int(label_item[1])] = ['I-' + label_item[2]] * (
                        int(label_item[1]) - int(label_item[0]) - 1)
                label_ids[int(label_item[0]) + 1:int(label_item[1])] = [label2id['I-' + label_item[2]]] * (
                        int(label_item[1]) - int(label_item[0]) - 1)

        assert len(text) == len(label_ids), 'text length is not equal to label ids length'
        assert len(text) == len(label_tag), 'text length is not equal to label tag length'
        label = {}
        label['label_tag'] = label_tag[:max_length]
        label['label_id'] = label_ids[:max_length]
        label['gold_label'] = labels[:max_length]
        return label
    @staticmethod
    def text2bios(text: str, labels: list, label2id: dict, max_length: int):
        label_tag = ['O'] * len(text)
        label_ids = [0] * len(text)

        for label_item in labels:
            assert text[int(label_item[0]):int(label_item[1])] == label_item[
                3], 'text:{} \n entity text : {} obtained by text[start:end] is not the same as label entity text:{} '.format(
                text,
                text[int(label_item[0]):int(label_item[1])], label_item[3])

            if 'B-' + label_item[2] not in label2id:
                label2id['B-' + label_item[2]] = len(label2id)
            if 'I-' + label_item[2] not in label2id:
                label2id['I-' + label_item[2]] = len(label2id)
            if 'S-' + label_item[2] not in label2id:
                label2id['S-' + label_item[2]] = len(label2id)

            if len(label_item[3]) == 1:
                label_tag[int(label_item[0])] = 'S-' + label_item[2]
                label_ids[int(label_item[0])] = label2id['S-' + label_item[2]]
            else:
                label_tag[int(label_item[0])] = 'B-' + label_item[2]
                label_ids[int(label_item[0])] = label2id['B-' + label_item[2]]
                label_tag[int(label_item[0]) + 1:int(label_item[1])] = ['I-' + label_item[2]] * (
                        int(label_item[1]) - int(label_item[0]) - 1)
                label_ids[int(label_item[0]) + 1:int(label_item[1])] = [label2id['I-' + label_item[2]]] * (
                        int(label_item[1]) - int(label_item[0]) - 1)

        assert len(text) == len(label_ids), 'text length is not equal to label ids length'
        assert len(text) == len(label_tag), 'text length is not equal to label tag length'
        label = {}
        label['label_tag'] = label_tag[:max_length]
        label['label_id'] = label_ids[:max_length]
        label['gold_label'] = labels[:max_length]
        return label
    @staticmethod
    def text2bioes(text: str, labels: list, label2id: dict, max_length: int):
        label_tag = ['O'] * len(text)
        label_ids = [0] * len(text)

        for label_item in labels:
            assert text[label_item[0]:label_item[1]] == label_item[
                3], 'entity text : {} obtained by text[start:end] is not the same as label entity text:{} '.format(
                text[label_item[0]:label_item[1]], label_item[3])

            if 'B-' + label_item[2] not in label2id:
                label2id['B-' + label_item[2]] = len(label2id)
            if 'I-' + label_item[2] not in label2id:
                label2id['I-' + label_item[2]] = len(label2id)
            if 'E-' + label_item[2] not in label2id:
                label2id['E-' + label_item[2]] = len(label2id)
            if 'S-' + label_item[2] not in label2id:
                label2id['S-' + label_item[2]] = len(label2id)

            if len(label_item[3]) == 1:
                label_tag[int(label_item[0])] = 'S-' + label_item[2]
                label_ids[int(label_item[0])] = label2id['S-' + label_item[2]]
            else:

                label_tag[int(label_item[0])] = 'B-' + label_item[2]
                label_ids[int(label_item[0])] = label2id['B-' + label_item[2]]

                label_tag[int(label_item[1]) - 1] = 'E-' + label_item[2]
                label_ids[int(label_item[1]) - 1] = label2id['E-' + label_item[2]]

                label_tag[int(label_item[0]) + 1:label_item[1] - 1] = ['I-' + label_item[2]] * (
                        int(label_item[1]) - int(label_item[0]) - 2)
                label_ids[int(label_item[0]) + 1:label_item[1] - 1] = [label2id['I-' + label_item[2]]] * \
                                                                      (int(label_item[1]) - int(label_item[0]) - 2)

        assert len(text) == len(label_ids), 'text length is not equal to label ids length'
        assert len(text) == len(label_tag), 'text length is not equal to label tag length'
        label = {}
        label['label_tag'] = label_tag[:max_length]
        label['label_id'] = label_ids[:max_length]
        label['gold_label'] = labels[:max_length]
        return label


    @staticmethod
    def get_entity_bio(seq, id2label=None):
        """Gets entities from sequence.
        note: BIO
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entity_bio(seq)
            #output
            [[0,2,'PER'], [ 3, 3,'LOC']]
        """
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
                chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx

                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return [[x[1], x[2] + 1, x[0]] for x in chunks]
    @staticmethod
    def get_entity_bios( seq, id2label):
        """Gets entities from sequence.
        note: BIOS
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
            # >>> get_entity_bios(seq)
            [[0,2,'PER'], [ 3, 3,'LOC']]
        """
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("S-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[2] = indx
                chunk[0] = tag.split('-')[1]
                chunks.append(chunk)
                chunk = (-1, -1, -1)
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return [[x[1], x[2] + 1, x[0]] for x in chunks]
    @staticmethod
    def get_entity_bieos( seq, id2label):
        """Gets entities from sequence.
        note: BIEOS
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            # >>> seq = ['B-PER', 'I-PER', 'E-PER', 'O', 'S-LOC']
            # >>> get_entity_bios(seq)
            [[0,2,'PER'], [ 3, 3,'LOC']]
        """
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("S-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[2] = indx
                chunk[0] = tag.split('-')[1]
                chunks.append(chunk)
                chunk = (-1, -1, -1)
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx
            elif tag.startswith('E-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx
                    chunks.append(chunk)
                    chunk = [-1, -1, -1]
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return [[x[1], x[2] + 1, x[0]] for x in chunks]

    @staticmethod
    def get_entities_from_str(seq, id2label):
        tag_set = set()
        for k in id2label.values():
            tag_set.add(k.upper().split('-')[0])

        tags = list(tag_set)
        tags.sort()
        if len(tags) == 3 and ''.join(tags) == 'BIO':
            return NERUtils.get_entity_bio(seq, id2label)
        elif len(tags) == 4 and ''.join(tags) == 'BIOS':
            return NERUtils.get_entity_bios(seq, id2label)
        elif len(tags) == 5 and ''.join(tags) == 'BEIOS':
            return NERUtils.get_entity_bieos(seq, id2label)
        elif len(tags) == 7:
            return NERUtils.get_entity_bieos(seq, id2label)
        else:
            raise Exception('{}, tags must be in the mode ["bio","bios","bieos"]'.format(tags))

    @staticmethod
    def get_entities(seq, id2label):
        if isinstance(seq[0], int) or isinstance(seq[0], np.int) or isinstance(seq[0], np.int64):
            seq = [id2label[x] for x in seq]
        return NERUtils.get_entities_from_str(seq, id2label)

    @staticmethod
    def get_entities_batch(seq_batch, length_batch, id2label):
        entities_seqs = []
        for seq, length in zip(seq_batch, length_batch):
            entities_seqs.append(NERUtils.get_entities(seq[:length], id2label))
        return entities_seqs




