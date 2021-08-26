# -*- coding: utf-8 -*-
# 分离后端函数，主要是为了同时兼容原生keras和tf.keras
# 通过设置环境变量TF_KERAS=1来切换tf.keras

import os, sys
from distutils.util import strtobool
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest, tf_inspect
from tensorflow.python.eager import tape
from tensorflow.python.ops.custom_gradient import _graph_mode_decorator

# 判断是tf.keras还是纯keras的标记
is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

if is_tf_keras:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    sys.modules['keras'] = keras
else:
    import keras
    import keras.backend as K

# 判断是否启用重计算（通过时间换空间）
do_recompute = strtobool(os.environ.get('RECOMPUTE', '0'))


def gelu_erf(x):
    """基于Erf直接计算的gelu函数
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    """基于Tanh近似计算的gelu函数
    """
    cdf = 0.5 * (
        1.0 + K.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3))))
    )
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


def piecewise_linear(t, schedule, from_zero=True):
    """分段线性函数
    其中schedule是形如{1000: 1, 2000: 0.1}的字典，
    表示 t ∈ [0, 1000]时，输出从0均匀增加至1，而
    t ∈ [1000, 2000]时，输出从1均匀降低到0.1，最后
    t > 2000时，保持0.1不变。
    """
    schedule = sorted(schedule.items())
    if from_zero and schedule[0][0] != 0:
        schedule = [(0, 0.0)] + schedule

    t = K.cast(t, K.floatx())
    x = (t * 0 + 1) * schedule[0][1]
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1.0 * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = (t * 0 + 1) * schedule[i][1]
        x = K.switch(t >= t_begin, x, x_begin)

    return x


def search_layer(inputs, name, exclude_from=None):
    """根据inputs和name来搜索层
    说明：inputs为某个层或某个层的输出；name为目标层的名字。
    实现：根据inputs一直往上递归搜索，直到发现名字为name的层为止；
         如果找不到，那就返回None。
    """
    if exclude_from is None:
        exclude_from = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude_from:
        return None
    else:
        exclude_from.add(layer)
        if isinstance(layer, keras.models.Model):
            model = layer
            for layer in model.layers:
                if layer.name == name:
                    return layer
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude_from)
                if layer is not None:
                    return layer


def sequence_masking(x, mask, value=0.0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    else:
        if K.dtype(mask) != K.dtype(x):
            mask = K.cast(mask, K.dtype(x))
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = K.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        return x * mask + value * (1 - mask)


def batch_gather(params, indices):
    """同tf旧版本的batch_gather
    """
    if K.dtype(indices)[:3] != 'int':
        indices = K.cast(indices, 'int32')

    try:
        return tf.gather(params, indices, batch_dims=K.ndim(indices) - 1)
    except Exception as e1:
        try:
            return tf.batch_gather(params, indices)
        except Exception as e2:
            raise ValueError('%s\n%s\n' % (e1.message, e2.message))


def pool1d(
    x,
    pool_size,
    strides=1,
    padding='valid',
    data_format=None,
    pool_mode='max'
):
    """向量序列的pool函数
    """
    x = K.expand_dims(x, 1)
    x = K.pool2d(
        x,
        pool_size=(1, pool_size),
        strides=(1, strides),
        padding=padding,
        data_format=data_format,
        pool_mode=pool_mode
    )
    return x[:, 0]


def divisible_temporal_padding(x, n):
    """将一维向量序列右padding到长度能被n整除
    """
    r_len = K.shape(x)[1] % n
    p_len = K.switch(r_len > 0, n - r_len, 0)
    return K.temporal_padding(x, (0, p_len))


def root_mean_square(x, axis=None, keepdims=False):
    """均方根，相当于模长的变体
    """
    return K.sqrt(K.mean(K.square(x), axis=axis, keepdims=keepdims))


def swish(x):
    """swish函数（这样封装过后才有 __name__ 属性）
    """
    return tf.nn.swish(x)


def leaky_relu(x, alpha=0.2):
    """leaky relu函数（这样封装过后才有 __name__ 属性）
    """
    return tf.nn.leaky_relu(x, alpha=alpha)


class Sinusoidal(keras.initializers.Initializer):
    """Sin-Cos位置向量初始化器
    来自：https://arxiv.org/abs/1706.03762
    """
    def __call__(self, shape, dtype=None):
        """Sin-Cos形式的位置向量
        """
        vocab_size, depth = shape
        embeddings = np.zeros(shape)
        for pos in range(vocab_size):
            for i in range(depth // 2):
                theta = pos / np.power(10000, 2. * i / depth)
                embeddings[pos, 2 * i] = np.sin(theta)
                embeddings[pos, 2 * i + 1] = np.cos(theta)
        return embeddings


def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：
        1. y_true和y_pred的shape一致，y_true的元素非0即1，
           1表示对应的类为目标类，0表示对应的类为非目标类；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
    pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


def symbolic(f):
    """恒等装饰器（兼容旧版本keras用）
    """
    return f


def graph_mode_decorator(f, *args, **kwargs):
    """tf 2.1与之前版本的传参方式不一样，这里做个同步
    """
    if tf.__version__ < '2.1':
        return _graph_mode_decorator(f, *args, **kwargs)
    else:
        return _graph_mode_decorator(f, args, kwargs)


def recompute_grad(call):
    """重计算装饰器（用来装饰Keras层的call函数）
    关于重计算，请参考：https://arxiv.org/abs/1604.06174
    """
    if not do_recompute:
        return call

    def inner(self, inputs, **kwargs):
        """定义需要求梯度的函数以及重新定义求梯度过程
        （参考自官方自带的tf.recompute_grad函数）
        """
        flat_inputs = nest.flatten(inputs)
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask', 'training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        def kernel_call():
            """定义前向计算
            """
            return call(self, inputs, **kwargs)

        def call_and_grad(*inputs):
            """定义前向计算和反向计算
            """
            if is_tf_keras:
                with tape.stop_recording():
                    outputs = kernel_call()
                    outputs = tf.identity(outputs)
            else:
                outputs = kernel_call()

            def grad_fn(doutputs, variables=None):
                watches = list(inputs)
                if variables is not None:
                    watches += list(variables)
                with tf.GradientTape() as t:
                    t.watch(watches)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(
                    outputs, watches, output_gradients=[doutputs]
                )
                del t
                return grads[:len(inputs)], grads[len(inputs):]

            return outputs, grad_fn

        if is_tf_keras:  # 仅在tf >= 2.0下可用
            outputs, grad_fn = call_and_grad(*flat_inputs)
            flat_outputs = nest.flatten(outputs)

            def actual_grad_fn(*doutputs):
                grads = grad_fn(*doutputs, variables=self.trainable_weights)
                return grads[0] + grads[1]

            watches = flat_inputs + self.trainable_weights
            watches = [tf.convert_to_tensor(x) for x in watches]
            tape.record_operation(
                call.__name__, flat_outputs, watches, actual_grad_fn
            )
            return outputs
        else:  # keras + tf >= 1.14 均可用
            return graph_mode_decorator(call_and_grad, *flat_inputs)

    return inner


# 给旧版本keras新增symbolic方法（装饰器），
# 以便兼容optimizers.py中的代码
K.symbolic = getattr(K, 'symbolic', None) or symbolic

custom_objects = {
    'gelu_erf': gelu_erf,
    'gelu_tanh': gelu_tanh,
    'gelu': gelu_erf,
    'root_mean_square': root_mean_square,
    'swish': swish,
    'leaky_relu': leaky_relu,
    'Sinusoidal': Sinusoidal,
    'multilabel_categorical_crossentropy': multilabel_categorical_crossentropy,
}

keras.utils.get_custom_objects().update(custom_objects)
