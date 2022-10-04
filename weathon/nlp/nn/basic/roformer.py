# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 17:29
# @Author  : LiZhen
# @FileName: reformer.py
# @github  : https://github.com/Lizhen0628
# @Description:
from abc import ABC

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from weathon.nlp.nn.configuration import RoFormerConfig
from weathon.nlp.nn.layer import RoFormerEncoder, RoFormerEmbeddings
from weathon.utils import TransformerUtils

ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = []
ROFORMER_PRETRAINED_MODEL_ARCHIVE_MAP = {}


class RoFormerPreTrainedModel(PreTrainedModel, ABC):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    config_class = RoFormerConfig
    pretrained_model_archive_map = ROFORMER_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = TransformerUtils.load_tf_weights_in_roformer
    base_model_prefix = "roformer"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RoFormerModel(RoFormerPreTrainedModel, ABC):
    def __init__(self, config):
        super().__init__(config)
        try:
            from transformers.modeling_bert import BertPooler
        except ImportError:
            from transformers.models.bert.modeling_bert import BertPooler

        self.config = config
        self.embeddings = RoFormerEmbeddings(config)
        self.encoder = RoFormerEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
            )
            encoder_hidden_shape = (encoder_batch_size,
                                    encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape,
                                                    device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask,
                                       self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_ids=input_ids,
                                           token_type_ids=token_type_ids,
                                           inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (
                      sequence_output,
                      pooled_output,
                  ) + encoder_outputs[
                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class RoFormer(RoFormerPreTrainedModel, ABC):
    """
    原始的RoFormer模型

    Args:
        config:
            模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练
        pooling (:obj:`str`, optional, defaults to "cls_with_pooler"):
            bert输出的池化方式，默认为"cls_with_pooler"，
            可选有["cls", "cls_with_pooler", "first_last_avg", "last_avg", "last_2_avg"]

    Reference:
        [1] https://github.com/ZhuiyiTechnology/roformer
        [2] https://github.com/JunnYu/RoFormer_pytorch
    """  # noqa: ignore flake8"

    def __init__(
            self,
            config,
            encoder_trained=True,
            pooling='cls_with_pooler'
    ):
        super(RoFormer, self).__init__(config)

        self.bert = RoFormerModel(config)
        self.pooling = pooling

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def mask_pooling(self, x: torch.Tensor, attention_mask=None):
        if attention_mask is None:
            return torch.mean(x, dim=1)
        return torch.sum(x * attention_mask.unsqueeze(2), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)

    def sequence_pooling(self, sequence_feature, attention_mask):

        if self.pooling == 'cls_with_pooler':
            return sequence_feature.pooler_output
        sequence_feature = sequence_feature.hidden_states
        if self.pooling == 'first_last_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[1]
        elif self.pooling == 'last_avg':
            sequence_feature = sequence_feature[-1]
        elif self.pooling == 'last_2_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[-2]
        elif self.pooling == 'cls':
            return sequence_feature[-1][:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(self.pooling))

        return self.mask_pooling(sequence_feature, attention_mask)

    def get_encoder_feature(self, encoder_output, attention_mask):
        if self.task == 'SequenceLevel':
            return self.sequence_pooling(encoder_output, attention_mask)
        elif self.task == 'TokenLevel':
            return encoder_output[-1]
        else:
            return encoder_output[-1][:, 0, :]

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True
        )

        encoder_feature = self.get_encoder_feature(outputs, attention_mask)

        encoder_feature = self.dropout(encoder_feature)
        out = self.classifier(encoder_feature)

        return out


class RoFormerForSequenceClassification(RoFormerPreTrainedModel, ABC):
    """
    基于RoFormer的文本分类模型

    Args:
        config:
            模型的配置对象

    Reference:
        [1] https://github.com/ZhuiyiTechnology/roformer
        [2] https://github.com/JunnYu/RoFormer_pytorch
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roformer = RoFormerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            **kwargs
    ):
        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits