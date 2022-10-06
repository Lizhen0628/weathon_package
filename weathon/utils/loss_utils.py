# -*- coding: utf-8 -*-
# @Time    : 2022/10/4 09:50
# @Author  : LiZhen
# @FileName: loss_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

import torch
from torch import nn
import torch.nn.functional as F
from weathon.nlp.factory.loss import all_losses_dict


class LossUtils:
    @staticmethod
    def get_loss(loss):
        """
        加载数据集

        Args:
            loss (:obj:`string` or :obj:`torch module`): 损失函数名或损失函数对象
        """  # noqa: ignore flake8"

        if isinstance(loss, str):
            loss = loss.lower()
            loss = loss.replace('_', '')
            return all_losses_dict[loss]()

        return loss

    @staticmethod
    def dice_loss(input, target, weight=None):
        """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
            Useful in dealing with unbalanced data

            input: [N, C]
            target: [N, ]
        """
        target_zeros = torch.zeros_like(target)
        target = torch.where(target < 0, target_zeros, target)
        prob = torch.softmax(input.transpose(1, 2), dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
        dsc_i = 1 - ((1 - prob) * prob) / ((1 - prob) * prob + 1)
        dice_loss = dsc_i.mean()
        return dice_loss

    @staticmethod
    def focal_loss( input, target, gamma=2, weight=None, ignore_index=-100):
        logpt = F.log_softmax(input.transpose(1, 2), dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** gamma * logpt
        loss = F.nll_loss(logpt, target, weight, ignore_index=ignore_index)
        return loss

    @staticmethod
    def label_smoothing_ce_loss(output, target, weight=None, epsilon=0.1, reduction='mean', ignore_index=-100):
        n_classes = output.size()[-1]
        if output.ndim == 3:
            log_preds = F.log_softmax(output.transpose(1, 2), dim=1)
        elif output.ndim == 2:
            log_preds = F.log_softmax(output, dim=1)

        if reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=1)
            if reduction == 'mean':
                loss = loss.mean()
        return loss * epsilon / n_classes + (1 - epsilon) * F.nll_loss(log_preds, target, weight=weight,
                                                                       reduction=reduction, ignore_index=ignore_index)

    @staticmethod
    def ce_loss(output, target, weight=None):
        if output.ndim == 3:  # 多见于序列标注任务
            return F.cross_entropy(output.float().transpose(1, 2), target, weight=weight, ignore_index=-100)
        elif output.ndim == 2:  # 多见于文本分类任务
            return F.cross_entropy(output.float(), target, weight=weight, ignore_index=-100)

    @staticmethod
    def binary_loss( output, target):
        return F.binary_cross_entropy_with_logits(output, target.float())

    @staticmethod
    def cosine_similarity_loss(first_vector, second_vector, label):
        return F.mse_loss(nn.Identity()(F.cosine_similarity(first_vector, second_vector)), label.view(-1))

    @staticmethod
    def triple_loss(anchor_vec, pos_vec, neg_vec, triplet_margin=5):
        distance_pos = F.pairwise_distance(anchor_vec, pos_vec, p=2)
        distance_neg = F.pairwise_distance(anchor_vec, neg_vec, p=2)

        return F.relu(distance_pos - distance_neg + triplet_margin).mean()

    @staticmethod
    def mrc_bce_loss(logits, labels, float_label_mask):
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        loss = (loss * float_label_mask).sum() / float_label_mask.sum()
        return loss

    @staticmethod
    def mrc_dice_loss(logits, labels, float_label_mask, smooth=1e-8, square_denominator=False):
        flat_input = logits.view(-1)
        flat_target = labels.view(-1)
        flat_input = torch.sigmoid(flat_input)

        if float_label_mask is not None:
            mask = float_label_mask.view(-1).float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask

        interection = torch.sum(flat_input * flat_target, -1)
        if not square_denominator:
            return 1 - ((2 * interection + smooth) / (flat_input.sum() + flat_target.sum() + smooth))
        else:
            return 1 - ((2 * interection + smooth) / (
                    torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target), -1) + smooth))
