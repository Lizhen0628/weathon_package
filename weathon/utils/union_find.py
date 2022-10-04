# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 10:40
# @Author  : LiZhen
# @FileName: union_find.py
# @github  : https://github.com/Lizhen0628
# @Description:

from typing import Set, List, Dict


class UnionFind(object):
    """并查集：查找不想交集合的数据结构

    是一种维护不相交集合的数据结构(连通分量)：
    1. 易于合并两个不连通的连通分量。
    2. 判断两个元素是否连通

    This implements the "weighted-quick-union-with-path-compression"
    union-find algorithm.  Only works if elements are immutable
    objects.


    Terms
    -----
    Component: 属于同一集合的元素

    Connected: 判断两个元素是否连通

    Union: 将两个components合并为一个

    Root: 表示集合祖先节点

    Find: 查找集合的祖先节点

    Parameters
    ----------
    elements : NoneType or container, optional, default: None
        The initial list of elements.

    Attributes
    ----------
    num_elements : int
        Number of elements.

    num_components : int
        Number of distjoint sets or components.

    Implements
    ----------
    __len__ : returns the number of elements.


    __contains__ : ``x in uf`` returns ``True`` if ``x`` is an element in ``uf``.

    __getitem__
        For ``uf`` an instance of ``UnionFind`` and ``i`` an integer,
        ``res = uf[i]`` returns the element stored in the ``i``-th index.
        If ``i`` is not a valid index an ``IndexError`` is raised.

    __setitem__
        For ``uf`` and instance of ``UnionFind``, ``i`` an integer and ``x``
        an immutable object, ``uf[i] = x`` changes the element stored at the
        ``i``-th index. If ``i`` is not a valid index an ``IndexError`` is
        raised.

    .. [1] http://algs4.cs.princeton.edu/lectures/

    """

    def __init__(self, elements=None):
        self.num_elements = 0  # current num of elements
        self.num_components = 0  # the number of disjoint sets or components
        self._next = 0  # next available id
        self._elements = []  # the elements
        self._indx = {}  # dict mapping elt -> index in _elts
        self._parent = []  # parent: for the internal tree structure
        self._size = []  # size of the component - correct only for roots

        if elements is None:
            elements = []
        for elem in elements:
            self.add(elem)

    def __repr__(self):
        return (
            '<UnionFind:\n\telements={},\n\tsize={},\n\tparent={},\nnum_elements={},num_components={}>'
                .format(
                self._elements,
                self._size,
                self._parent,
                self.num_elements,
                self.num_components,
            ))

    def __len__(self):
        return self.num_elements

    def __contains__(self, x):
        return x in self._indx

    def __getitem__(self, index):
        if index < 0 or index >= self._next:
            raise IndexError('index {} is out of bound'.format(index))
        return self._elements[index]

    def __setitem__(self, index, x):
        if index < 0 or index >= self._next:
            raise IndexError('index {} is out of bound'.format(index))
        self._elements[index] = x

    def add(self, elem) -> None:
        """
        Add a single disjoint element.
        Args:
            elem:
        Returns: None
        """

        if elem in self:
            return
        self._elements.append(elem)
        self._indx[elem] = self._next
        self._parent.append(self._next)
        self._size.append(1)
        self._next += 1
        self.num_elements += 1
        self.num_components += 1

    def find(self, elem) -> int:
        """Find the root of the disjoint set containing the given element.
        Args:
            elem:
        Returns:
            The (index of the) root.

        Raise ValueError:
            If the given element is not found.
        """
        if elem not in self._indx:
            raise ValueError('{} is not an element'.format(elem))

        p = self._indx[elem]
        while p != self._parent[p]:
            # path compression
            q = self._parent[p]
            self._parent[p] = self._parent[q]
            p = q
        return p

    def connected(self, x, y):
        """
        Return whether the two given elements belong to the same component.
        Args:
            x:
            y:
        Returns:
            True if x and y are connected, false otherwise.
        """
        return self.find(x) == self.find(y)

    def union(self, x, y) -> None:
        """
        Merge the components of the two given elements into one.
        Args:
            x:
            y:
        Returns:
        """

        # Initialize if they are not already in the collection
        for elt in [x, y]:
            if elt not in self:
                self.add(elt)

        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self._size[xroot] < self._size[yroot]:
            self._parent[xroot] = yroot
            self._size[yroot] += self._size[xroot]
        else:
            self._parent[yroot] = xroot
            self._size[xroot] += self._size[yroot]
        self.num_components -= 1

    def component(self, x) -> Set:
        """Find the connected component containing the given element.
        Args:
            x:

        Returns:set

        """
        if x not in self:
            raise ValueError('{} is not an element'.format(x))
        root = self.find(x)
        return set((elt for elt in self._elements if self.find(elt) == root))

    def components(self) -> List[Set]:
        """
        Return the list of connected components.
        Returns:A list of sets.
        """

        components_dict = {}
        for elt in self._elements:
            root = self.find(elt)
            components_dict.setdefault(root, set()).add(elt)
        return list(components_dict.values())

    def component_mapping(self) -> Dict:
        """
        Return a dict mapping elements to their components.
        Returns:
            A dict with the semantics: `elt -> component contianing elt`.
        """
        components_dict = {}
        mapping = {}
        for elt in self._elements:
            root = self.find(elt)
            components_dict.setdefault(root, set()).add(elt)
            mapping[elt] = components_dict[root]
        return mapping


if __name__ == '__main__':
    uf = UnionFind(list('abcdefghij'))
    uf.union('e', 'd')
    uf.union('d', 'i')
    uf.union('g', 'f')
    uf.union('j', 'e')
    uf.union('c', 'b')
    uf.union('i', 'j')
    uf.union('f', 'a')
    uf.union('h', 'c')
    uf.union('g', 'b')
    uf.union('a', 'b')
    uf.union('g', 'h')
    print(uf.connected('a', 'g'))
    print(uf.component('a'))
    print(uf.components())
    print(uf.component_mapping())
