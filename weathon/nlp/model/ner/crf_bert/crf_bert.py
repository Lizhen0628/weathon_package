from weathon.nlp.nn.basic import BertForTokenClassification
from weathon.nlp.nn.layer import CRF


class CrfBert(BertForTokenClassification):
    """
    基于BERT + CRF的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(CrfBert, self).__init__(config, encoder_trained)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
