import json

from weathon.nlp.base import BaseVocab
from typing import Union, List, Set


class LabelVocab(BaseVocab):

    def __init__(self, initial_labels: Union[List[str], Set[str]] = None):
        super().__init__()
        self.id2label = {}
        self.label2id = {}

        self.initial_labels = initial_labels
        for label in self.initial_labels:
            self.add(label)

    def add(self, label: str, cnt: int = 1) -> int:
        if label in self.label2id:
            idx = self.label2id[label]
        else:
            idx = len(self.id2label)
            self.id2label[idx] = label
            self.label2id[label] = idx
        return idx

    def get_id(self, label):
        try:
            return self.label2id[label]
        except KeyError:
            raise Exception("Invalid label!")

    def get_label(self, idx):
        try:
            return self.id2label[idx]
        except KeyError:
            raise Exception("Invalid id!")

    def convert_labels_to_ids(self, labels: Union[List[str], Set[str]]) -> List[int]:
        return [self.get_id(label) for label in labels]

    def recover_from_ids(self, ids: List[int], stop_id=None):
        return [self.get_label(idx) for idx in ids]

    def recover_id2label(self):
        return {idx: label for label, idx in self.label2id}

    def save(self, output_path: str = './label2id.json') -> None:
        with open(output_path, mode='w', encoding='utf8') as f:
            json.dump(obj=self.label2id, fp=f, ensure_ascii=False)

    def load(self, save_path='./label2id.json') -> None:
        with open(save_path, mode='r', encoding='utf8') as f:
            self.label2id = json.load(fp=f)
        self.id2label = self.recover_id2label()
