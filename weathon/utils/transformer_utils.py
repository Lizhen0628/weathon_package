# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 17:02
# @Author  : LiZhen
# @FileName: transformer_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

import re
import os
import torch
import logging
import numpy as np
from transformers import AutoModel, AutoTokenizer, BertTokenizer

logger = logging.getLogger(__name__)


class TransformerUtils:

    @staticmethod
    def download_from_huggingface(model_name, root_path):
        """
        1. 下载可能会失败，重试即可
        2. 程序会缓存之前下载过的内容
        Examples:
            TransformerUtils.download_from_huggingface('clue/albert_chinese_small','/data/lizhen/pretrained_models')
        """
        path = os.path.join(root_path, model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(path)

        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(path)
        print(os.path.abspath(path), 'done.')

    @staticmethod
    def load_tf_weights_in_nezha(model, config, tf_checkpoint_path):
        """Load tf checkpoints in a pytorch model."""
        try:

            import tensorflow as tf
        except ImportError:
            logger.error(
                "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                "https://www.tensorflow.org/install/ for installation instructions."
            )
            raise

        tf_path = os.path.abspath(tf_checkpoint_path)
        logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
        # Load weights from TF model
        init_vars = tf.train.list_variables(tf_path)
        names = []
        arrays = []
        for name, shape in init_vars:
            # logger.info("Loading TF weight {} with shape {}".format(name, shape))
            array = tf.train.load_variable(tf_path, name)
            names.append(name)
            arrays.append(array)

        for name, array in zip(names, arrays):
            name = name.split("/")
            # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
            # which are not required for using pretrained model
            if any(
                    n in ["adam_v", "adam_m", "lamb_m", "lamb_v", "AdamWeightDecayOptimizer",
                          "AdamWeightDecayOptimizer_1",
                          "global_step", "good_steps", "loss_scale", 'bad_steps']
                    for n in name
            ):
                logger.info("Skipping {}".format("/".join(name)))
                continue
            pointer = model
            for m_name in name:
                if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = re.split(r"_(\d+)", m_name)
                else:
                    scope_names = [m_name]
                if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                    pointer = getattr(pointer, "bias")
                elif scope_names[0] == "output_weights":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "squad":
                    pointer = getattr(pointer, "classifier")
                else:
                    try:
                        pointer = getattr(pointer, scope_names[0])
                    except AttributeError:
                        logger.info("Skipping {}".format("/".join(name)))
                        continue
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]
            if m_name[-11:] == "_embeddings":
                pointer = getattr(pointer, "weight")
            elif m_name == "kernel":
                array = np.transpose(array)
            try:
                assert (
                        pointer.shape == array.shape
                ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
        return model

    @staticmethod
    def load_tf_weights_in_roformer(model, config, tf_checkpoint_path):
        """Load tf checkpoints in a pytorch model."""
        try:
            import tensorflow as tf
        except ImportError:
            logger.error(
                "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                "https://www.tensorflow.org/install/ for installation instructions."
            )
            raise
        tf_path = os.path.abspath(tf_checkpoint_path)
        logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
        # Load weights from TF model
        init_vars = tf.train.list_variables(tf_path)
        names = []
        arrays = []
        for name, shape in init_vars:
            logger.info("Loading TF weight {} with shape {}".format(name, shape))
            array = tf.train.load_variable(tf_path, name)
            names.append(name.replace("bert", "roformer"))
            arrays.append(array)

        for name, array in zip(names, arrays):
            name = name.split("/")
            # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
            # which are not required for using pretrained model
            if any(n in [
                "adam_v", "adam_m", "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1", "global_step"
            ] for n in name):
                logger.info("Skipping {}".format("/".join(name)))
                continue
            pointer = model
            for m_name in name:
                if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = re.split(r"_(\d+)", m_name)
                else:
                    scope_names = [m_name]
                if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                    pointer = getattr(pointer, "bias")
                elif scope_names[0] == "output_weights":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "squad":
                    pointer = getattr(pointer, "classifier")
                else:
                    try:
                        pointer = getattr(pointer, scope_names[0])
                    except AttributeError:
                        logger.info("Skipping {}".format("/".join(name)))
                        continue
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]
            if m_name[-11:] == "_embeddings":
                pointer = getattr(pointer, "weight")
            elif m_name == "kernel":
                array = np.transpose(array)
            try:
                assert (
                        pointer.shape == array.shape
                ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
        return model
