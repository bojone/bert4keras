# -*- coding: utf-8 -*-
# 训练相关

from .backend import keras, K

# 等价于 from keras.optimizers import Optimizer
globals()['Optimizer'] = keras.optimizers.__dict__['Optimizer']


class PiecewiseLinearLearningRate(Optimizer):
    """分段线性学习率优化器
    传入优化器，然后将优化器的学习率改为分段线性的，
    返回修改后的优化器。主要是用来实现warmup等功能。
    """
    def __init__(self, optimizer, schedule=None, **kwargs):
        super(PiecewiseLinearLearningRate, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.schedule = schedule

    @property
    def lr(self):
        return self.optimizer.lr

    @property
    def iterations(self):
        return self.optimizer.iterations

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
            self.optimizer.lr = self.calculate_lr(self.iterations,
                                                  self.schedule)
        self.updates = self.optimizer.get_updates(loss, params)
        return self.updates
