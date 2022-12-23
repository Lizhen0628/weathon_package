# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 19:54
# @Author  : LiZhen
# @FileName: optimizer_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:
from torch.optim import Optimizer
from weathon.nlp.factory.optimizer import all_optimizers_dict, AdamW


class OptimizerUtils:

    @staticmethod
    def get_optimizer(optimizer, module, lr=False, params=None):
        if params is None:
            params_ = (p for p in module.parameters() if p.requires_grad)
        else:
            params_ = params

        if isinstance(optimizer, str):
            optimizer = all_optimizers_dict[optimizer](params_)
        elif type(optimizer).__name__ == 'type' and issubclass(optimizer, Optimizer):
            optimizer = optimizer(params_)
        elif isinstance(optimizer, Optimizer):
            if params is not None:
                optimizer.param_groups = params
        else:
            raise ValueError("The optimizer type does not exist")

        if lr is not False:
            for param_groups_ in optimizer.param_groups:
                param_groups_['lr'] = lr

        return optimizer

    @staticmethod
    def get_default_optimizer(module, module_name='bert', **kwargs):
        module_name = module_name.lower()

        if module_name == 'bert':
            return OptimizerUtils.get_default_bert_optimizer(module, **kwargs)
        elif module_name == 'crf_bert':
            return OptimizerUtils.get_default_crf_bert_optimizer(module, **kwargs)
        else:
            raise ValueError("The default optimizer does not exist")

    @staticmethod
    def get_default_bert_optimizer(
            module,
            lr: float = 3e-5,
            eps: float = 1e-6,
            correct_bias: bool = True,
            weight_decay: float = 1e-3,
    ):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = all_optimizers_dict['adamw'](optimizer_grouped_parameters,
                                                 lr=lr,
                                                 eps=eps,
                                                 correct_bias=correct_bias,
                                                 weight_decay=weight_decay)
        return optimizer

    @staticmethod
    def get_default_crf_bert_optimizer(
            module,
            lr: float = 2e-5,
            crf_lr: float = 2e-3,
            eps: float = 1e-6,
            correct_bias: bool = True,
            weight_decay: float = 1e-2,
    ):
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(module.bert.named_parameters())
        crf_param_optimizer = list(module.crf.named_parameters())
        linear_param_optimizer = list(module.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay, 'lr': lr},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': lr},
            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay, 'lr': crf_lr},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': crf_lr},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay, 'lr': crf_lr},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': crf_lr}
        ]
        optimizer = all_optimizers_dict['adamw'](optimizer_grouped_parameters)

        return optimizer

    @staticmethod
    def get_transformer_parameters(model, lr: float = 3e-5, weight_decay: float = 1e-3):
        no_decay = ["bias", "LayerNorm.weight"]
        group_parameters = [
            {"params": [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay, "lr": lr},
            {"params": [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, "lr": lr},
        ]
        return group_parameters

    @staticmethod
    def get_classifier_parameters(model, lr: float = 1e-3, weight_decay: float = 1e-3):
        no_decay = ["bias", "LayerNorm.weight"]
        group_parameters = [
            {"params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay, "lr": lr},
            {"params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, "lr": lr},
        ]
        return group_parameters

    @staticmethod
    def get_tc_transformer_optimizer_adamw(
            model,
            transformer_lr: float = 3e-5,
            classifier_lr: float = 1e-3,
            weight_decay: float = 1e-3,
            eps: float = 1e-6
    ):
        group_parameters = OptimizerUtils.get_transformer_parameters(model, transformer_lr, weight_decay) + \
                           OptimizerUtils.get_classifier_parameters(model, classifier_lr, weight_decay)

        optimizer = AdamW(group_parameters, eps=eps)
        return optimizer

    @staticmethod
    def get_w2ner_model_optimizer(
            dl_module,
            lr: float = 1e-3,
            bert_lr: float = 5e-6,
            weight_decay=0.0
    ):
        bert_params = set(dl_module.bert.parameters())
        other_params = list(set(dl_module.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in dl_module.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': bert_lr,
             'weight_decay': weight_decay},
            {'params': [p for n, p in dl_module.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': bert_lr,
             'weight_decay': weight_decay},
            {'params': other_params,
             'lr': lr,
             'weight_decay': weight_decay},
        ]

        optimizer = all_optimizers_dict['adamw'](params, lr=lr, weight_decay=weight_decay)

        return optimizer
