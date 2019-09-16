#! -*- coding: utf-8 -*-
# 自定义层

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import Model


"""
gelu有两个实现版本，
一是利用Erf直接计算，二是利用Tanh做近似，
两者会有一点差异。
官方早期放出的代码是用Erf函数实现的，
但当前的官方代码已经改为了Tanh版本。
"""
gelu_version = 1


def gelu_erf(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    cdf = 0.5 * (1.0 + K.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
    return x * cdf


if gelu_version == 1:
    gelu = gelu_erf
else:
    gelu = gelu_tanh


def add_seq_mask(x, mask, mode=0, axis=None, heads=1):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    mode: 如果是0，则直接乘以mask；
          如果是1，则在padding部分减去一个大正数。
    axis: 序列所在轴，默认为1；
    heads: 相当于batch这一维要被重复的次数。
    """
    if mask is None or mode not in [0, 1]:
        return x
    else:
        if heads is not 1:
            mask = K.expand_dims(mask, 1)
            mask = K.tile(mask, (1, heads, 1))
            mask = K.reshape(mask, (-1, K.shape(mask)[2]))
        if axis is None:
            axis = 1
        if axis == -1:
            axis = K.ndim(x) - 1
        assert axis > 0, 'axis muse be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 0:
            return x * mask
        else:
            return x - (1 - mask) * 1e12


class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        for u in layer.updates:
            if not hasattr(self, '_updates'):
                self._updates = []
            if u not in self._updates:
                self._updates.append(u)
        return outputs


class MultiHeadAttention(OurLayer):
    """多头注意力机制
    """
    def __init__(self, heads, head_size, key_size=None, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = heads * head_size
        self.key_size = key_size if key_size else head_size

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads)
        self.k_dense = Dense(self.key_size * self.heads)
        self.v_dense = Dense(self.out_dim)
        self.o_dense = Dense(self.out_dim)

    def call(self, inputs, mask=None):
        """实现多头注意力
        注意：这个mask输入是对Attention矩阵的mask。
             如果mask是None或True，则忽略；如果mask是True，
             则自动mask掉未来信息（做语言模型用）；如果mask
             是一个张量，则直接用这个张量来mask。
        """
        q, k, v = inputs[:3]
        v_mask = q_mask = None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # 转为三阶张量
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.head_size))
        # Attention
        a = K.batch_dot(qw, kw, [2, 2]) / np.sqrt(self.key_size)
        a = add_seq_mask(a, v_mask, 1, -1, self.heads)
        if (mask is not None) and (mask is not False):
            if mask is True:
                ones = K.ones_like(a[:1])
                mask = (ones - tf.matrix_band_part(ones, -1, 0)) * 1e12
                a = a - mask
            else:
                a = a - (1 - mask) * 1e12
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [2, 1])
        o = K.reshape(o, (-1, self.heads, K.shape(q)[1], self.head_size))
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.reuse(self.o_dense, o)
        o = add_seq_mask(o, q_mask, 0)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class LayerNormalization(Layer):
    """实现基本的Layer Norm，只保留核心运算部分
    """
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = K.epsilon() * K.epsilon()

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        shape = (input_shape[-1], )
        self.gamma = self.add_weight(shape=shape,
                                     initializer='ones',
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer='zeros',
                                    name='beta')

    def call(self, inputs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs *= self.gamma
        outputs += self.beta
        return outputs


class PositionEmbedding(Layer):
    """定义位置Embedding，这里的Embedding是可训练的。
    """
    def __init__(self, input_dim, output_dim, merge_mode='add', **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(name='embeddings',
                                          shape=(self.input_dim,
                                                 self.output_dim),
                                          initializer='zeros')

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embeddings = self.embeddings[:seq_len]
        pos_embeddings = K.expand_dims(pos_embeddings, 0)
        pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])
        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        else:
            return K.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.v_dim, )


class FeedForward(OurLayer):
    """FeedForward层，其实就是两个Dense层的叠加
    """
    def __init__(self, units, activation='relu', **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        self.dense_1 = Dense(self.units, activation=self.activation)
        self.dense_2 = Dense(output_dim)

    def call(self, inputs):
        x = self.reuse(self.dense_1, inputs)
        x = self.reuse(self.dense_2, x)
        return x


class EmbeddingDense(Layer):
    """运算跟Dense一致，只不过kernel用Embedding层的embedding矩阵
    """
    def __init__(self, embedding_layer, activation='softmax', **kwargs):
        super(EmbeddingDense, self).__init__(**kwargs)
        self.kernel = K.transpose(embedding_layer.embeddings)
        self.activation = activation

    def build(self, input_shape):
        super(EmbeddingDense, self).build(input_shape)
        self.bias = self.add_weight(name='bias',
                                    shape=(K.int_shape(self.kernel)[1],),
                                    initializer='zeros')

    def call(self, inputs):
        outputs = K.dot(inputs, self.kernel)
        outputs = K.bias_add(outputs, self.bias)
        outputs = Activation(self.activation).call(outputs)
        return outputs
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (K.int_shape(self.kernel)[1],)
