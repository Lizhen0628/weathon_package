# -*- coding: utf-8 -*-
# @Time    : 2022/10/6 10:39
# @Author  : LiZhen
# @FileName: textrank.py
# @github  : https://github.com/Lizhen0628
# @Description:

import sys
from collections import defaultdict


class TextRank:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, start, end, weight=1):
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        ws = defaultdict(float)
        out_sum = defaultdict(float)

        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            out_sum[n] = sum((e[2] for e in out), 0.0)

        sorted_keys = sorted(self.graph.keys())
        for x in range(10):
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / out_sum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s

        min_rank, max_rank = sys.float_info[0], sys.float_info[3]
        for w in ws.values():
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in ws.items():
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws
