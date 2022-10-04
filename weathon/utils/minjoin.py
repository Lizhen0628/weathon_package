# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 10:08
# @Author  : LiZhen
# @FileName: minjoin.py
# @github  : https://github.com/Lizhen0628
# @Description:

import mmh3
from typing import *
import Levenshtein


class Mmh3:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, s: str) -> int:
        return mmh3.hash(s, self.seed)


class Patitioner(object):

    def __init__(self, seg: int, w: int, hash_fns: List[Callable[[str], int]], part_size_rate: float = 0.5):
        self.seg = seg
        self.w = w
        self.hash_fns = hash_fns
        self.rate = part_size_rate

    def _find_anchors(self, s: str) -> List[List[int]]:
        outs = []
        L = int(len(s) / self.seg)
        z = max(1, int(L / 2))
        # z = int((len(s) - self.w + 1 - self.seg) / (2 * self.seg + 2))

        for hash_fn in self.hash_fns:
            out = [0]
            h = [hash_fn(s[i:i + self.w]) for i in range(len(s) - self.w + 1)]

            i = 0
            while i < len(h):
                d = 1
                while d <= z:
                    if (i + d >= len(h) or h[i] < h[i + d]) and (i - d < 0 or h[i] < h[i - d]):
                        d += 1
                    else:
                        break

                if d == z + 1:
                    out.append(i)
                i += d

            out.append(len(s))
            outs.append(out)
        return outs

    def _find_part(self, s: str, anchors_list: List[List[int]]) -> List[Tuple[str, int]]:
        T = int(len(s) * self.rate / self.seg)
        out = set()
        for anchors in anchors_list:
            for i in range(1, len(anchors)):
                length = anchors[i] - anchors[i - 1]
                if length > T:
                    out.add((s[anchors[i - 1]:anchors[i]], anchors[i - 1]))
                    # if len(out) == 1:
                    #     for i in range(0, len(s), T):
                    #         out.add((s[i * T:(i + 1) * T], i * T))

        return list(out)

    def partition(self, s: str) -> List[Tuple[str, int]]:
        anchors = self._find_anchors(s)
        return self._find_part(s, anchors)


class MinJoin(object):

    def __init__(self, seeds: List[int], max_length: int, sim: float, ngram: int):
        self.seeds = seeds
        self.sim = sim
        self.ngram = ngram
        self.seg = max(int(max_length * (1 - sim)), 2)
        self.partitioner = Patitioner(self.seg, self.ngram, [Mmh3(seed) for seed in self.seeds])

    def process(self, contents: List[str]) -> Dict[str, List[str]]:
        sub_tables: Dict[str, List[Tuple[str, int]]] = {}
        for s in contents:
            parts = self.partitioner.partition(s)
            for sub, index in parts:
                sub_tables.setdefault(sub, []).append((s, index))

        res: Dict[str, List[str]] = {}
        for _, values in sub_tables.items():
            values.sort(key=lambda x: len(x[0]))

            for i in range(len(values)):
                left, left_index = values[i]
                left_n = len(left)
                for j in range(i + 1, len(values)):
                    right, right_index = values[j]
                    if right == left or right in res.get(left, []):
                        continue
                    right_n = len(right)
                    # length = max(left_n,right_n)
                    length = right_n
                    thresh = (1 - self.sim) * length
                    if right_n - left_n > thresh:
                        break
                    if abs(left_index - right_index) + abs(left_n - left_index - right_n + right_index) > thresh:
                        continue
                    if Levenshtein.distance(left, right) > thresh:
                        continue
                    res.setdefault(left, []).append(right)
                    res.setdefault(right, []).append(left)
        return res


def generate_edges(text_ids_mapping: Dict[str, Set[str]], min_join: MinJoin) -> Set[Tuple[str, str]]:
    res = min_join.process(list(text_ids_mapping.keys()))
    edges = set()

    for key, value_list in res.items():
        for left_id in text_ids_mapping.get(key, []):
            for value in value_list:
                for right_id in text_ids_mapping.get(value, []):
                    if left_id < right_id:
                        edges.add((left_id, right_id))

    for values in text_ids_mapping.values():
        values = list(values)
        values.sort()
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                edges.add((values[i], values[j]))
    return edges


if __name__ == '__main__':
    min_join = MinJoin(seeds=[100, 200], max_length=35, sim=0.8, ngram=1)
    texts = [
        "福建琯溪蜜柚新鲜现摘水果2-3斤/个 微甜",
        "福建琯溪蜜柚新鲜现摘水果2-3斤/个",
        "新鲜现摘水果2-3斤/个 微甜",
        "补气益血 特级新疆灰枣1000g+50g包装随机",
        "补气益血特级新疆灰枣1000g+50g包装随机,",
        "补气特级新疆灰枣1000g+50g包装随机"
    ]
    result = min_join.process(texts)
    print(result)
