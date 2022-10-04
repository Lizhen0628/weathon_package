# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 20:24
# @Author  : LiZhen
# @FileName: token_classification.py
# @github  : https://github.com/Lizhen0628
# @Description:


import torch
import numpy as np
from weathon.utils.ner_utils import NERUtils
from weathon.nlp.base import BasePredictor
from weathon.nlp.predictor.sequence_classification import SequenceClassificationPredictor



class TokenClassificationPredictor(SequenceClassificationPredictor):
    """
    字符分类任务的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):

        super(TokenClassificationPredictor, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'TokenLevel'

    def predict_one_sample(self, text='', topk=1, return_label_name=True, return_proba=False):
        """
        单样本预测

        Args:
            topk: 
            return_proba: 
            return_label_name: 
            text (:obj:`string`): 输入文本
        """  # noqa: ignore flake8"

        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs)

        preds = logit.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        preds = preds[0][1:]
        preds = preds[:len(text)]

        # tags = [self.id2cat[x] for x in preds]
        label_entities = NERUtils.get_entities(preds, self.id2cat)

        entities = set()
        for entity_ in label_entities:
            entities.add(text[entity_[1]: entity_[2] + 1] + '-' + entity_[0])

        entities = []
        for entity_ in label_entities:
            entities.append({
                "start_idx": entity_[1],
                "end_idx": entity_[2],
                "entity": text[entity_[1]: entity_[2] + 1],
                "type": entity_[0]
            })

        return entities


class BiaffineNERPredictor(object):
    """
    Biaffine命名实体的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id
    ):
        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text
    ):
        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        input_ids = self.tokenizer.sequence_to_ids(tokens)
        input_ids, input_mask, segment_ids = input_ids

        zero = [0 for i in range(self.tokenizer.max_seq_len)]
        span_mask = [input_mask for i in range(sum(input_mask))]
        span_mask.extend([zero for i in range(sum(input_mask), self.tokenizer.max_seq_len)])
        span_mask = np.array(span_mask)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'span_mask': span_mask
        }

        return features, token_mapping

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text=''
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
        """  # noqa: ignore flake8"

        features, token_mapping = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            scores = torch.argmax(self.module(**inputs), dim=-1)[0].to(torch.device('cpu')).numpy().tolist()

        entities = []
        for start in range(len(scores)):
            for end in range(start, len(scores[start])):
                if scores[start][end] > 0:
                    if end - 1 > token_mapping[-1][-1]:
                        break
                    if token_mapping[start - 1][0] <= token_mapping[end - 1][-1]:
                        entitie_ = {
                            "start_idx": token_mapping[start - 1][0],
                            "end_idx": token_mapping[end - 1][-1],
                            "entity": text[token_mapping[start - 1][0]: token_mapping[end - 1][-1] + 1],
                            "type": self.id2cat[scores[start][end]]
                        }

                        if entitie_['entity'] == '':
                            continue

                        entities.append(entitie_)

        return entities


class BIONERPredictor(object):
    """
    BIO模式的字符分类任务的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id,
            markup='bio'
    ):
        self.markup = markup

        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text
    ):
        input_ids = self.tokenizer.sequence_to_ids(text)
        input_ids, input_mask, segment_ids = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids
        }
        return features

    def _convert_to_vanilla_ids(
            self,
            text
    ):
        tokens = self.tokenizer.tokenize(text)
        length = len(tokens)
        input_ids = self.tokenizer.sequence_to_ids(tokens)

        features = {
            'input_ids': input_ids,
            'length': length if length < self.tokenizer.max_seq_len else self.tokenizer.max_seq_len,
        }
        return features

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text=''
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
        """  # noqa: ignore flake8"

        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs)

        preds = logit.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        preds = preds[0][1:]
        preds = preds[:len(text)]

        # tags = [self.id2cat[x] for x in preds]
        label_entities = NERUtils.get_entities(preds, self.id2cat)

        entities = set()
        for entity_ in label_entities:
            entities.add(text[entity_[1]: entity_[2] + 1] + '-' + entity_[0])

        entities = []
        for entity_ in label_entities:
            entities.append({
                "start_idx": entity_[1],
                "end_idx": entity_[2],
                "entity": text[entity_[1]: entity_[2] + 1],
                "type": entity_[0]
            })

        return entities


class CRFNERPredictor(object):
    """
    +CRF模式的字符分类任务的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id,
            markup='bio'
    ):
        self.markup = markup

        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text
    ):
        input_ids = self.tokenizer.sequence_to_ids(text)
        input_ids, input_mask, segment_ids = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids
        }
        return features

    def _convert_to_vanilla_ids(
            self,
            text
    ):
        tokens = self.tokenizer.tokenize(text)
        length = len(tokens)
        input_ids = self.tokenizer.sequence_to_ids(tokens)

        features = {
            'input_ids': input_ids,
            'length': length if length < self.tokenizer.max_seq_len else self.tokenizer.max_seq_len,
        }
        return features

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text=''
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
        """  # noqa: ignore flake8"

        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs)

        tags = self.module.crf.decode(logit, inputs['attention_mask'])
        tags = tags.squeeze(0)

        preds = tags.detach().cpu().numpy().tolist()
        preds = preds[0][1:]
        preds = preds[:len(text)]

        tags = [self.id2cat[x] for x in preds]
        label_entities = NERUtils.get_entities(preds, self.id2cat)

        entities = set()
        for entity_ in label_entities:
            entities.add(text[entity_[1]: entity_[2] + 1] + '-' + entity_[0])

        entities = []
        for entity_ in label_entities:
            entities.append({
                "start_idx": entity_[1],
                "end_idx": entity_[2],
                "entity": text[entity_[1]: entity_[2] + 1],
                "type": entity_[0]
            })

        return entities


class GlobalPointerNERPredictor(object):
    """
    GlobalPointer命名实体识别的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id
    ):
        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text
    ):

        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        input_ids = self.tokenizer.sequence_to_ids(tokens)
        input_ids, input_mask, segment_ids = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids
        }

        return features, token_mapping

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text='',
            threshold=0
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
            threshold (:obj:`float`, optional, defaults to 0): 预测的阈值
        """  # noqa: ignore flake8"

        features, token_mapping = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            scores = self.module(**inputs)[0].cpu()

        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf

        entities = []
        for category, start, end in zip(*np.where(scores > threshold)):
            if end - 1 > token_mapping[-1][-1]:
                break
            if token_mapping[start - 1][0] <= token_mapping[end - 1][-1]:
                entitie_ = {
                    "start_idx": token_mapping[start - 1][0],
                    "end_idx": token_mapping[end - 1][-1],
                    "entity": text[token_mapping[start - 1][0]: token_mapping[end - 1][-1] + 1],
                    "type": self.id2cat[category]
                }

                if entitie_['entity'] == '':
                    continue

                entities.append(entitie_)

        return entities


class SpanNERPredictor(object):
    """
    span模式的命名实体识别的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id
    ):
        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text
    ):
        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        input_ids = self.tokenizer.sequence_to_ids(tokens)
        input_ids, input_mask, segment_ids = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
        }

        return features, token_mapping

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text=''
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
        """  # noqa: ignore flake8"

        features, token_mapping = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            start_logits, end_logits = self.module(**inputs)
            start_scores = torch.argmax(start_logits[0].cpu(), -1).numpy()[1:]
            end_scores = torch.argmax(end_logits[0].cpu(), -1).numpy()[1:]

        entities = []
        for index_, s_l in enumerate(start_scores):
            if s_l == 0:
                continue

            if index_ > token_mapping[-1][-1]:
                break

            for jndex_, e_l in enumerate(end_scores[index_:]):

                if index_ + jndex_ > token_mapping[-1][-1]:
                    break

                if s_l == e_l:
                    entitie_ = {
                        "start_idx": token_mapping[index_][0],
                        "end_idx": token_mapping[index_ + jndex_][-1],
                        "type": self.id2cat[s_l],
                        "entity": text[token_mapping[index_][0]: token_mapping[index_ + jndex_][-1] + 1]
                    }
                    entities.append(entitie_)
                    break

        return entities



class PromptMLMPredictor(BasePredictor):

    def __init__(
        self,
        *args,
        prompt,
        prompt_mode='postfix',
        prefix_special_token_num=1,
        **kwargs
    ):
        super(PromptMLMPredictor, self).__init__(*args, **kwargs)
        self.prompt = prompt
        self.prompt_mode = prompt_mode
        self.prefix_special_token_num = prefix_special_token_num

    def _convert_to_transfomer_ids(
        self,
        text
    ):

        seq = self.tokenizer.tokenize(text)

        if self.prompt_mode == 'postfix':
            start_mask_position = len(seq) + self.prefix_special_token_num + self.prompt.index("[MASK]")
        else:
            start_mask_position = self.prefix_special_token_num + self.prompt.index("[MASK]")

        mask_position = [
            start_mask_position + index
            for index in range(len(self.tokenizer.tokenize(list(self.cat2id.keys())[0])))
        ]

        input_ids = self.tokenizer.sequence_to_ids(seq, self.prompt)
        input_ids, input_mask, segment_ids = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'mask_position': np.array(mask_position)
        }

        return features

    def predict_one_sample(
        self,
        text='',
        topk=1,
        return_label_name=True,
        return_proba=False
    ):
        if topk is None:
            topk = len(self.cat2id) if len(self.cat2id) > 2 else 1

        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs).cpu().numpy()

        # [label_num, label_length]
        labels_ids = np.array(
            [self.tokenizer.vocab.convert_tokens_to_ids(
                self.tokenizer.tokenize(_cat)) for _cat in self.cat2id])

        preds = np.ones(shape=[len(labels_ids)])

        label_length = len(self.tokenizer.tokenize(list(self.cat2id.keys())[0]))

        for index in range(label_length):
            preds *= logit[index, labels_ids[:, index]]

        preds = torch.Tensor(preds)
        preds = preds.reshape(1, -1)

        probs, indices = preds.topk(topk, dim=1, sorted=True)

        preds = []
        probas = []
        for pred_, proba_ in zip(indices.cpu().numpy()[0], probs.cpu().numpy()[0].tolist()):

            if return_label_name:
                pred_ = self.id2cat[pred_]

            preds.append(pred_)

            if return_proba:
                probas.append(proba_)

        if return_proba:
            return list(zip(preds, probas))

        return preds
