# -*- coding: utf-8 -*-
# 分离后端函数，主要是为了同时兼容原生keras和tf.keras
# 通过设置环境变量TF_KERAS=1来切换tf.keras

import os
from distutils.util import strtobool
import numpy as np
import tensorflow as tf


if strtobool(os.environ.get('TF_KERAS', '0')):
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
else:
    import keras
    import keras.backend as K


def get_all_attributes(something):
    """获取类下的所有属性和方法
    """
    return {
        name: getattr(something, name)
        for name in dir(something) if name[:2] != '__' and name[-2:] != '__'
    }

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
    verision = verision.lower()
    assert version in ['erf', 'tanh'], 'gelu version must be erf or tanh'
    if verision == 'erf':
        keras.utils.get_custom_objects()['gelu'] = gelu_erf
    else:
        keras.utils.get_custom_objects()['gelu'] = gelu_tanh


custom_objects = {
    'gelu_erf': gelu_erf,
    'gelu_tanh': gelu_tanh,
    'gelu': gelu_erf,
}

keras.utils.get_custom_objects().update(custom_objects)
