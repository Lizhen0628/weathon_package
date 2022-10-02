import dgl

from weathon.nlp.base import BaseTokenizer, BaseVocab
from transformers import BertTokenizer, AutoTokenizer
from typing import Union, List, Set,Tuple


class TextLevelGCNTokenizer(BaseTokenizer):
    """
    文本编码器，用于对文本进行图编码

    Args:
        vocab: 词典类对象，用于实现文本分词和ID化
        max_seq_len (:obj:`int`): 预设的文本最大长度
        graph: 图类对象，用于生成子图

    """  # noqa: ignore flake8"

    def __init__(self, vocab: Union[BaseVocab, BertTokenizer, AutoTokenizer], max_seq_len: int, graph):
        super(TextLevelGCNTokenizer, self).__init__(vocab, max_seq_len)
        self.graph = graph
        self.tokenizer_type = 'graph'

    def sequence_to_graph(self, sequence:Union[str,List[str]]) -> Tuple[List[int],List[int],dgl.DGLHeteroGraph]:
        if type(sequence) == str:
            sequence = self.tokenize(sequence)

        sequence = self.vocab.convert_tokens_to_ids(sequence)
        if len(sequence) == 0:
            sequence = [0]

        node_ids = list(set(sequence))
        local_token2id = dict(zip(node_ids, range(len(node_ids))))

        sub_graph = dgl.graph([])

        # 节点信息
        sub_graph.add_nodes(len(node_ids))

        # 边和权信息
        local_edges, edge_ids = self.graph.get_sequence_graph(sequence, local_token2id)

        srcs, dsts = zip(*local_edges)
        sub_graph.add_edges(srcs, dsts)

        return node_ids, edge_ids, sub_graph
