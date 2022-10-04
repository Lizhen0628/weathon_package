from weathon.nlp.nn.layer import CRF
from weathon.nlp.nn.basic import BertForTokenClassification


class CrfBert(BertForTokenClassification):
    """
    基于BERT + CRF 的命名实体模型

    Args:
        config: 
            模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练
    """  # noqa: ignore flake8"

    def __init__(
            self,
            config,
            encoder_trained=True
    ):
        super(CrfBert, self).__init__(config, encoder_trained)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
