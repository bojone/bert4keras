# -*- coding: utf-8 -*-
# 训练相关

from .backend import keras, K

# 等价于 from keras.optimizers import Optimizer
locals()['Optimizer'] = keras.optimizers.Optimizer
# 等价于 from keras.utils import get_custom_objects
locals()['get_custom_objects'] = keras.utils.get_custom_objects


class OptimizerWrapper(Optimizer):
    """优化器包装
    """
    def __init__(self, optimizer, **kwargs):
        super(OptimizerWrapper, self).__init__(**kwargs)
        self.optimizer = optimizer

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @property
    def iterations(self):
        return self.optimizer.iterations

    def get_updates(self, loss, params):
        return self.optimizer.get_updates(loss, params)

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
    传入优化器，然后将优化器的学习率改为分段线性的，
    返回修改后的优化器。主要是用来实现warmup等功能。
    """
    def __init__(self, optimizer, schedule=None, **kwargs):
        super(PiecewiseLinearLearningRate, self).__init__(optimizer, **kwargs)
        self.schedule = {int(i): j for i, j in schedule.items()}

    def calculate_lr(self, iterations, schedule):
        """根据schedule定义分段线性函数
        """
        schedule = sorted(schedule.items())
        if schedule[0][0] != 0:
            schedule = [(0, 0.)] + schedule
        lr = K.constant(schedule[0][1], dtype='float32')
        t = K.cast(iterations, 'float32')
        for i in range(len(schedule)):
            if i == len(schedule) - 1:
                x = K.constant(schedule[i][1], dtype='float32')
            else:
                dx = schedule[i + 1][1] - schedule[i][1]
                dt = schedule[i + 1][0] - schedule[i][0]
                k = 1. * dx / dt
                t0 = schedule[i][0]
                x0 = schedule[i][1]
                x = x0 + k * (t - t0)
            lr = K.switch(t >= t0, x, lr)
        return lr

    def get_updates(self, loss, params):
        if self.schedule is not None:
            self.optimizer.learning_rate = self.calculate_lr(
                iterations=self.iterations,
                schedule=self.schedule,
            )
        self.updates = self.optimizer.get_updates(loss, params)
        return self.updates

    def get_config(self):
        config = {'schedule': self.schedule}
        base_config = super(PiecewiseLinearLearningRate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'PiecewiseLinearLearningRate': PiecewiseLinearLearningRate,
}

get_custom_objects().update(custom_objects)
