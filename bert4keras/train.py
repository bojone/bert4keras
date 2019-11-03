# -*- coding: utf-8 -*-
# 训练相关（不适用于tf.keras中的OptimizerV2类优化器）

from bert4keras.backend import keras, K
from bert4keras.backend import get_all_attributes
from bert4keras.backend import piecewise_linear
import re


class OptimizerWrapper(keras.optimizers.Optimizer):
    """优化器包装，主要为了方便做一些修改原有优化器的工作
    """
    def __init__(self, optimizer, **kwargs):
        super(OptimizerWrapper, self).__init__(**kwargs)
        self.optimizer = optimizer
        self._optimizer_attributes = []
        for k, v in get_all_attributes(self.optimizer).items():
            if k not in dir(self):
                setattr(self, k, v)
                self._optimizer_attributes.append(k)

    def get_updates(self, loss, params):
        for k in self._optimizer_attributes:
            setattr(self.optimizer, k, getattr(self, k))
        self.updates = self.optimizer.get_updates(loss, params)
        self.weights = self.optimizer.weights
        return self.updates

    def get_config(self):
        config = {
            'optimizer': keras.optimizers.serialize(self.optimizer),
        }
        base_config = super(OptimizerWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        optimizer = keras.optimizers.deserialize(config.pop('optimizer'))
        return cls(optimizer, **config)


class PiecewiseLinearLearningRate(OptimizerWrapper):
    """分段线性学习率优化器
    传入优化器，然后将优化器的学习率改为分段线性的。
    其中schedule是形如{1000: 1, 2000: 0.1}的字典，
    表示0～1000步内学习率线性地从零增加到100%，然后
    1000～2000步内线性地降到10%，2000步以后保持10%.
    """
    def __init__(self, optimizer, schedule=None, **kwargs):
        super(PiecewiseLinearLearningRate, self).__init__(optimizer, **kwargs)
        self.schedule = {int(i): j for i, j in schedule.items()}
        factor = piecewise_linear(self.iterations, self.schedule)
        self.learning_rate = self.learning_rate * factor

    def get_config(self):
        config = {'schedule': self.schedule}
        base_config = super(PiecewiseLinearLearningRate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GradientAccumulation(OptimizerWrapper):
    """梯度累积优化器
    将steps_per_update步的梯度平均起来，然后才更新模型。
    """
    def __init__(self, optimizer, steps_per_update=1, **kwargs):
        super(GradientAccumulation, self).__init__(optimizer, **kwargs)
        self.steps_per_update = steps_per_update
        # 判断是否要更新的标记
        self.cond = K.equal(self.iterations % self.steps_per_update, 0)
        # 用学习率来决定是否更新，不更新即学习率为0
        self.learning_rate = K.switch(self.cond, self.learning_rate, 0.)
        # 滑动平均量在非更新期内不要动
        for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
            if hasattr(self, attr):
                value = K.switch(self.cond, getattr(self, attr), 1. - 1e-7)
                setattr(self, attr, value)

    def get_gradients(self, loss, params):
        if hasattr(self, 'accum_grads'):
            return [ag / self.steps_per_update for ag in self.accum_grads]
        else:
            return super(GradientAccumulation,
                         self).get_gradients(loss, params)

    def get_updates(self, loss, params):
        # 获取梯度
        grads = self.get_gradients(loss, params)
        # 定义累积梯度
        self.accum_grads = [
            K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params
        ]
        accum_updates = [
            K.update(ag, K.switch(self.cond, ag * 0, ag + g))
            for g, ag in zip(grads, self.accum_grads)
        ]
        # 继承原更新
        self.optimizer.get_gradients = self.get_gradients
        super(GradientAccumulation, self).get_updates(loss, params)
        self.updates.extend(accum_updates)
        self.weights.extend(self.accum_grads)
        return self.updates

    def get_config(self):
        config = {'steps_per_update': self.steps_per_update}
        base_config = super(GradientAccumulation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def add_weight_decay_into(model, weight_decay_rate, exclude_from=None):
    """往模型加入权重衰减（权重衰减不等价于L2正则）
    """
    if exclude_from is None:
        exclude_from = []

    def need_to_do_weight_decay(w):
        for n in exclude_from:
            if re.search(n, w.name):
                return False
        return True

    weight_decay_updates = []
    factor = 1 - weight_decay_rate
    for w in model.trainable_weights:
        if need_to_do_weight_decay(w):
            weight_decay_updates.append(K.update(w, w * factor))

    model.add_update(weight_decay_updates)


custom_objects = {
    'PiecewiseLinearLearningRate': PiecewiseLinearLearningRate,
    'GradientAccumulation': GradientAccumulation,
}

keras.utils.get_custom_objects().update(custom_objects)
