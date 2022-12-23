# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 21:00
# @Author  : LiZhen
# @FileName: task.py
# @github  : https://github.com/Lizhen0628
# @Description:

import time
import torch
import warnings
from typing import List
from numpy import inf
from abc import abstractmethod
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from weathon.utils import EMA, ScheduleUtils
from weathon.nlp.base.dataset import BaseDataset


class BaseTask(object):
    """
    所有Task类的基类，封装Task类通用的方法和属性

    Args:
        module: 深度学习模型
        optimizer: 训练模型使用的优化器名或者优化器对象
        loss_function: 训练模型使用的损失函数名或损失函数对象
        class_num (:obj:`int` or :obj:`None`, optional, defaults to None): 标签数目
        scheduler (:obj:`class`, optional, defaults to None): scheduler对象
        n_gpu (:obj:`int`, optional, defaults to 1): GPU数目
        device (:obj:`class`, optional, defaults to None): torch.device对象，当device为None时，会自动检测是否有GPU
        cuda_device (:obj:`int`, optional, defaults to 0): GPU编号，当device为None时，根据cuda_device设置device
        ema_decay (:obj:`int` or :obj:`None`, optional, defaults to None): EMA的加权系数
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

    def __init__(self, model, optimizer, loss_function, class_num=None, scheduler=None, n_gpu=1, device=None,
                 cuda_device=0, ema_decay=None, **kwargs):
        self.fit_counter = 0
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.class_num = class_num
        self.scheduler = scheduler

        self.n_gpu = n_gpu

        self.device = device

        if self.device is None:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                self.device = "cpu"

        self.model.to(self.device)

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.ema_decay = ema_decay
        if self.ema_decay:
            self.ema = EMA(self.model.parameters(), decay=self.ema_decay)

    def fit(self, train_data, validation_data=None, batch_size=32, epochs=1, gradient_accumulation_steps=1,
            early_stop: int = inf, **kwargs):
        """
        训练方法
        Args:
            train_data (:obj:`ark_nlp dataset`): 训练的batch文本
            validation_data (:obj:`ark_nlp dataset`): 验证的batch文本
            batch_size (:obj:`int`, optional, defaults to 32): batch大小
            epochs (:obj:`int`, optional, defaults to 1): 训练轮数
            gradient_accumulation_steps (:obj:`int`, optional, defaults to 1): 梯度累计数
            early_stop
            **kwargs (optional): 其他可选参数
        """  # noqa: ignore flake8"

        train_generator = self._train_begin(train_data, validation_data, batch_size,epochs=epochs, shuffle=True, **kwargs)

        for epoch in range(epochs):
            self._epoch_begin(**kwargs)  # module.train(), 重置 epoch级日志：epoch_loss,epoch_evaluation, epoch_step
            for step, inputs in enumerate(train_generator):
                self._step_begin(epoch, step, inputs, **kwargs)  # 重置 step级日志记录
                # input处理和设备转移
                inputs = self._get_module_inputs_on_train(inputs, **kwargs)

                # forward
                outputs = self.model(**inputs)

                # 计算损失
                logits, loss = self._get_train_loss(inputs, outputs, **kwargs)

                # loss backword
                loss = self._loss_backward(inputs, outputs, logits, loss, **kwargs)
                self._step_criterion_record(loss,**kwargs)
                if (step + 1) % gradient_accumulation_steps == 0:
                    # optimize
                    self._optimize_step(inputs, outputs, logits, loss, **kwargs)

                # setp evaluate
                self._step_end(step, inputs, outputs, loss, **kwargs)

            self._epoch_end(epoch, **kwargs)

            if validation_data is not None:
                self.evaluate(validation_data, **kwargs)

        self._train_end(**kwargs)

    def evaluate(self, validation_data, evaluate_batch_size=16, **kwargs):
        """
        验证方法

        Args:
            validation_data (:obj:`ark_nlp dataset`): 训练的batch文本
            evaluate_batch_size (:obj:`int`, optional, defaults to 32): 验证阶段batch大小
            **kwargs (optional): 其他可选参数
        """  # noqa: ignore flake8"

        self.evaluate_logs = dict()
        evaluate_generator = self._evaluate_begin(validation_data, evaluate_batch_size, shuffle=False, **kwargs)

        with torch.no_grad():
            for step, inputs in enumerate(evaluate_generator):
                inputs = self._get_module_inputs_on_eval(inputs, **kwargs)

                # forward
                outputs = self.model(**inputs)

                self._evaluate_step_end(inputs, outputs, **kwargs)
        self._evaluate_end(**kwargs)

    def _train_collate_fn(self, batch):
        """ 训练集
        接收一个batch数据，将数据转化成tensor
        """
        return default_collate(batch)

    def _evaluate_collate_fn(self, batch):
        """验证集
        接收一个batch数据，将数据转化成tensor
        """
        return default_collate(batch)

    def _train_begin(self, train_data: BaseDataset, validation_data: BaseDataset, batch_size: int, epochs: int,
                     shuffle: bool = True, warmup_proportion: float = None,
                     num_workers: int = 0, train_to_device_cols: List = None, **kwargs):
        """训练开始前的一些准备操作：
            1. 准备训练集数据标签和标签id的映射关系、标签数量
            2. tensor使用哪些列的特征数据
            3. 准备训练集生成器
            4. scheduler
            6. 定义训练日志变量，记录全局训练指标数据：global_step 和 global_loss
            7. 返回训练集生成器
        """
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = train_data.cat2id

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        self.train_to_device_cols = train_to_device_cols if train_to_device_cols else train_data.to_device_cols
        train_generator = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                     collate_fn=self._train_collate_fn)
        self.train_generator_length = len(train_generator)

        self.scheduler = self._prepare_scheduler(self.train_generator_length, warmup_proportion,
                                                 epochs) if warmup_proportion else None

        self.optimizer.zero_grad()

        # 初始化 global_step 和 global_loss
        self._train_begin_record()

        return train_generator

    def _evaluate_begin(self, validation_data, batch_size, shuffle, num_workers=0, evaluate_to_device_cols=None,
                        **kwargs):
        self.model.eval()
        self.evaluate_to_device_cols = evaluate_to_device_cols if evaluate_to_device_cols else validation_data.to_device_cols
        evaluate_generator = DataLoader(validation_data, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=num_workers,
                                        collate_fn=self._evaluate_collate_fn)

        if self.ema_decay:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())

        self._evaluate_begin_record(**kwargs)
        return evaluate_generator

    def _train_begin_record(self):
        """ 训练开始之前的初始化操作
        1. 全局日志初始化
        """
        self.logs = dict()
        self.logs['global_step'] = 0
        self.logs['global_loss'] = 0

        self.logs["best_epoch"] = 0
        self.logs["not_improved_count"] = 0
        self.logs["best_criterion"] = inf

    def _evaluate_begin_record(self, **kwargs):
        self.evaluate_logs = dict()
        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_example'] = 0

    def _get_train_loss(self, inputs, outputs, **kwargs):
        """to override
        计算训练损失
        """
        if type(outputs) == tuple:
            if len(outputs) > 2:
                logits, loss, *_ = outputs
            else:
                logits, loss = outputs
        else:
            logits = outputs
            # 计算损失
            loss = self._compute_loss(inputs, logits, **kwargs)
        return logits, loss

    def _get_evaluate_loss(self, inputs, outputs, verbose=True, **kwargs):
        if type(outputs) == tuple:
            if len(outputs) > 2:
                logits, loss, *_ = outputs
            else:
                logits, loss = outputs
        else:
            logits = outputs
            # 计算损失
            loss = self._compute_loss(inputs, logits, **kwargs)
        return logits, loss

    def _epoch_begin(self, **kwargs):
        self.model.train()
        # TODO ： 是否需要梯度置零
        # self.model.zero_grad()

        # 重置 epoch级别 指标记录
        self._epoch_begin_record(**kwargs)

    def _prepare_scheduler(self, train_generator_length, warmup_proportion, epochs, **kwargs):
        if warmup_proportion:
            num_training_steps = train_generator_length * epochs
            num_warmup_steps = int(warmup_proportion * num_training_steps)
            scheduler = ScheduleUtils.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=num_warmup_steps,
                                                                      num_training_steps=num_training_steps)
            return scheduler
        return None

    def _epoch_begin_record(self, **kwargs):
        """

        """
        self.logs['epoch_loss'] = 0
        # 占位作用，子类仅使用单个指标进行评价，则直接使用该字段即可
        self.logs['epoch_evaluation'] = 0
        self.logs['epoch_step'] = 0

    @abstractmethod
    def _compute_loss(self, inputs, logits, verbose=True, **kwargs):
        loss = self.loss_function(logits, inputs['label_ids'])
        return loss

    def _step_criterion_record(self, loss, **kwargs):
        self.logs["global_loss"] += loss.item()
        self.logs["epoch_loss"] += loss.item()

    def _loss_backward(self, inputs, outputs, logits, loss, gradient_accumulation_steps: int = 1, loss_cut: float = 0.0,
                       **kwargs):
        """训练梯度反向传播"""
        # 如果GPU数量大于1
        loss = loss.mean() if self.n_gpu > 1 else loss
        # 如果使用了梯度累积，除以累积的轮数
        loss = loss / gradient_accumulation_steps
        loss = torch.where(loss > float(loss_cut), loss, torch.zeros_like(loss))
        loss.backward()
        return loss

    def _prepare_optimize(self, **kwargs):
        pass

    def _optimize_step(self, inputs, outputs, logits, loss, grad_clip=None, **kwargs):

        # 梯度裁剪
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        # 更新权值
        self.optimizer.step()

        if self.ema_decay:
            self.ema.update(self.model.parameters())

        # 更新学习率
        if self.scheduler:
            self.scheduler.step()

        # 清空梯度
        self.optimizer.zero_grad()
        self._optimize_record(inputs, outputs, logits, loss, **kwargs)

    def _optimize_record(self, inputs, outputs, logits, loss, **kwargs):
        self.logs['global_step'] += 1
        self.logs['epoch_step'] += 1

    def _step_end(self, step, inputs, outputs, loss, verbose=True, show_step=100, **kwargs):
        if verbose and (step + 1) % show_step == 0:
            print(
                f"[{step}/{self.train_generator_length}],train loss is:{self.logs['epoch_loss'] / self.logs['epoch_step']:.6f}")

    def _evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['eval_loss'] += loss.item()

        self.evaluate_logs['eval_example'] += len(inputs['label_ids'])
        self.evaluate_logs['eval_step'] += 1

    def _epoch_end(self, epoch, verbose=True, **kwargs):
        if verbose:
            print(f"epoch:[{epoch}],train loss is:{self.logs['epoch_loss'] / self.logs['epoch_step']:.6f} \n")

    def _train_end(self, **kwargs):
        pass

    def _evaluate_end(self, evaluate_save=False, save_module_path=None,is_evaluate_print=True, **kwargs):
        if is_evaluate_print:
            print(f"test loss is:{self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step']:.6f}")
        if evaluate_save:
            if save_module_path is None:
                prefix = './checkpoint/' + str(self.model.__class__.__name__) + '_'
                save_module_path = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

            torch.save(self.model.state_dict(), save_module_path)

        if self.ema_decay:
            self.ema.restore(self.model.parameters())

        self._evaluate_end_record()

    def _evaluate_end_record(self, **kwargs):
        pass

    def _get_module_inputs_on_train(self, inputs, **kwargs):
        for col in self.train_to_device_cols:

            if type(inputs[col]) is torch.Tensor:
                inputs[col] = inputs[col].to(self.device)
            else:
                warnings.warn(f"The {col} is not Tensor.\n")
        return inputs

    def _get_module_inputs_on_eval(self, inputs, **kwargs):
        for col in self.evaluate_to_device_cols:
            if type(inputs[col]) is torch.Tensor:
                inputs[col] = inputs[col].to(self.device)
            else:
                warnings.warn(f"The {col} is not Tensor.\n")

        return inputs

    def _get_module_label_on_train(self,inputs,**kwargs):
        pass

    def _get_module_label_on_eval(self,inputs,**kwargs):
        pass

    def _step_begin(self,epoch,step, inputs, **kwargs):
        pass

    def _evaluate_step_begin(self,step,inputs,**kwargs):
        pass
