# -*- coding: utf-8 -*-
# 分离后端函数，主要是为了同时兼容原生keras和tf.keras
# 通过设置环境变量TF_KERAS=1来切换tf.keras

import os
from distutils.util import strtobool
import numpy as np
import tensorflow as tf


# 判断是tf.keras还是纯keras的标记
is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

if is_tf_keras:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
else:
    import keras
    import keras.backend as K


def gelu_erf(x):
    """基于Erf直接计算的gelu函数
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    """基于Tanh近似计算的gelu函数
    """
    cdf = 0.5 * (1.0 + K.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
    return x * cdf


def set_gelu(version):
    """设置gelu版本
    """
    version = version.lower()
    assert version in ['erf', 'tanh'], 'gelu version must be erf or tanh'
    if version == 'erf':
        keras.utils.get_custom_objects()['gelu'] = gelu_erf
    else:
        keras.utils.get_custom_objects()['gelu'] = gelu_tanh


def piecewise_linear(t, schedule):
    """分段线性函数
    其中schedule是形如{1000: 1, 2000: 0.1}的字典，
    表示 t ∈ [0, 1000]时，输出从0均匀增加至1，而
    t ∈ [1000, 2000]时，输出从1均匀降低到0.1，最后
    t > 2000时，保持0.1不变。
    """
    schedule = sorted(schedule.items())
    if schedule[0][0] != 0:
        schedule = [(0, 0.)] + schedule

    x = K.constant(schedule[0][1], dtype=K.floatx())
    t = K.cast(t, K.floatx())
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1. * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = K.constant(schedule[i][1], dtype=K.floatx())
        x = K.switch(t >= t_begin, x, x_begin)

    return x


custom_objects = {
    'gelu_erf': gelu_erf,
    'gelu_tanh': gelu_tanh,
    'gelu': gelu_erf,
    'swish': tf.nn.swish,
    'leaky_relu': tf.nn.leaky_relu,
}

keras.utils.get_custom_objects().update(custom_objects)
