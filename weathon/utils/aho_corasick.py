# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 17:21
# @Author  : LiZhen
# @FileName: aho_corasick.py
# @github  : https://github.com/Lizhen0628
# @Description:
# 参考资料：
# 1. [ac自动机算法详解](https://blog.csdn.net/bestsort/article/details/82947639)
# 2. [AC自动机](https://blog.csdn.net/weixin_40317006/article/details/81327188)


from collections import defaultdict



class Node(object):
    """
    node
    """

    def __init__(self, str='', is_root=False):
        self._next_p = {}
        self.fail = None
        self.is_root = is_root
        self.str = str
        self.parent = None

    def __iter__(self):
        return iter(self._next_p.keys())

    def __getitem__(self, item):
        return self._next_p[item]

    def __setitem__(self, key, value):
        _u = self._next_p.setdefault(key, value)
        _u.parent = self

    def __repr__(self):
        return "<Node object '%s' at %s>" % \
               (self.str, object.__repr__(self)[1:-1].split('at')[-1])

    def __str__(self):
        return self.__repr__()


class AhoCorasick(object):
    """
    Ac object
    """

    def __init__(self, *words):
        self.words = list(set(words))
        self.words.sort(key=lambda x: len(x))
        self._root = Node(is_root=True)
        self._node_meta = defaultdict(set)  # 存放的是以字符结尾的词，以及词的长度
        self._node_all = [(0, self._root)]  # 记录字符的层级信息
        self._initialize()
        self._make()

    def _initialize(self):
        self._search_char_related_words()
        for word in self.words:
            self._node_append(word)
        self._node_all.sort(key=lambda x: x[0])  # 按照层级信息排序,以便层次遍历

    def _node_append(self, keyword: str):
        """build trie"""
        assert len(keyword) > 0, "keyword length is zero"
        cur_root = self._root
        for char_idx, char in enumerate(keyword):
            node = Node(char)
            if char in cur_root:
                pass
            else:
                cur_root[char] = node
                self._node_all.append((char_idx + 1, cur_root[char]))
            if char_idx >= 1:
                for related_word in self.char_related_words[char]:
                    if keyword[:char_idx + 1].endswith(related_word):
                        self._node_meta[id(cur_root[char])].add((related_word, len(related_word)))
            cur_root = cur_root[char]
        else:
            if cur_root != self._root:
                self._node_meta[id(cur_root)].add((keyword, len(keyword)))

    def _search_char_related_words(self):
        self.char_related_words = {}  # 存放了和字符所有相关联的词
        for word in self.words:
            for char in word:
                self.char_related_words.setdefault(char, set())
                self.char_related_words[char].add(word)

    def _make(self):
        """
        build ac tree
        :return:
        """
        for _level, node in self._node_all:  # 第一层的fail节点一定是root
            if node == self._root or _level <= 1:
                node.fail = self._root
            else:
                _node = node.parent.fail
                while True:
                    if node.str in _node:
                        node.fail = _node[node.str]
                        break
                    else:
                        if _node == self._root:
                            node.fail = self._root
                            break
                        else:
                            _node = _node.fail

    def search(self, content, with_index=False):
        result = set()
        node = self._root
        index = 0
        for i in content:
            while 1:
                if i not in node:
                    if node == self._root:
                        break
                    else:
                        node = node.fail
                else:
                    for keyword, keyword_len in self._node_meta.get(id(node[i]), set()):
                        if not with_index:
                            result.add(keyword)
                        else:
                            result.add((keyword, (index - keyword_len + 1, index + 1)))
                    node = node[i]
                    break
            index += 1
        return result


if __name__ == '__main__':
    ac = AhoCorasick("阿傍",
                     "阿谤",
                     "阿保",
                     "阿保之功",
                     "阿保之劳",
                     "阿本郎",
                     "阿鼻",
                     "阿鼻地狱",
                     "阿鼻鬼",
                     "阿鼻叫唤",
                     "阿鼻狱",
                     "阿比",
                     "阿比让", "abc", 'abe', 'acdabd', 'bdf', 'df', 'f', 'ac', 'cd', 'cda')
    print(ac.search('acda阿鼻狱bdf', True))
