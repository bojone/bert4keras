# -*- coding: utf-8 -*-
# 分离后端函数，主要是为了同时兼容原生keras和tf.keras
# 通过设置环境变量TF_KERAS=1来切换tf.keras

import os
from distutils.util import strtobool


if strtobool(os.environ.get('TF_KERAS', '0')):
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
else:
    import keras
    import keras.backend as K
