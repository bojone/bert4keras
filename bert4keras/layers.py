#! -*- coding: utf-8 -*-
# 自定义层

import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.backend import search_layer
from bert4keras.backend import sequence_masking
from bert4keras.backend import pool1d
from bert4keras.backend import divisible_temporal_padding
from bert4keras.snippets import is_string
from keras import initializers, activations
from keras.layers import *


if keras.__version__[-2:] != 'tf' and keras.__version__ < '2.3':

    class Layer(keras.layers.Layer):
        """重新定义Layer，赋予“层中层”功能
        （仅keras 2.3以下版本需要）
        """
        def __setattr__(self, name, value):
            if isinstance(value, keras.layers.Layer):
                if not hasattr(self, '_layers'):
                    self._layers = []
                if value not in self._layers:
                    self._layers.append(value)
            super(Layer, self).__setattr__(name, value)

        @property
        def trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            if trainable:
                trainable_weights = super(Layer, self).trainable_weights[:]
                for l in getattr(self, '_layers', []):
                    trainable_weights += l.trainable_weights
                return trainable_weights
            else:
                return []

        @property
        def non_trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            non_trainable_weights = super(Layer, self).non_trainable_weights[:]
            for l in getattr(self, '_layers', []):
                if trainable:
                    non_trainable_weights += l.non_trainable_weights
                else:
                    non_trainable_weights += l.weights
            return non_trainable_weights


class ZeroMasking(Layer):
    """啥都不做，就是加上mask
    """
    def call(self, inputs):
        self._output_mask = K.cast(K.greater(inputs, 0), K.floatx())
        return inputs

    @property
    def output_mask(self):
        return self._output_mask


class MultiHeadAttention(Layer):
    """多头注意力机制
    """
    def __init__(self,
                 heads,
                 head_size,
                 key_size=None,
                 pool_size=None,
                 kernel_initializer='glorot_uniform',
                 max_relative_position=None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = heads * head_size
        self.key_size = key_size or head_size
        self.pool_size = pool_size or 1
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.max_relative_position = max_relative_position

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(units=self.key_size * self.heads,
                             kernel_initializer=self.kernel_initializer)
        self.k_dense = Dense(units=self.key_size * self.heads,
                             kernel_initializer=self.kernel_initializer)
        self.v_dense = Dense(units=self.out_dim,
                             kernel_initializer=self.kernel_initializer)
        self.o_dense = Dense(units=self.out_dim,
                             kernel_initializer=self.kernel_initializer)

        if self.max_relative_position is not None:
            if self.head_size != self.key_size:
                raise ValueError('head_size must be equal to key_size ' +
                                 'while use relative position embeddings.')

            def initializer(shape, dtype=None):
                vocab_size, depth = shape
                embeddings = np.zeros(shape)
                for pos in range(vocab_size):
                    for i in range(depth // 2):
                        theta = pos / np.power(10000, 2. * i / depth)
                        embeddings[pos, 2 * i] = np.sin(theta)
                        embeddings[pos, 2 * i + 1] = np.cos(theta)
                return embeddings

            shape = (2 * self.max_relative_position + 1, self.head_size)
            self.relative_embeddings = self.add_weight(name='relative_embeddings',
                                                       shape=shape,
                                                       initializer=initializer,
                                                       trainable=False)

    def call(self, inputs, q_mask=None, v_mask=None, a_mask=None):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        a_mask: 对attention矩阵的mask。
                不同的attention mask对应不同的应用。
        """
        q, k, v = inputs[:3]
        if a_mask:
            if len(inputs) == 3:
                a_mask = 'history_only'
            else:
                a_mask = inputs[3]
        if q_mask is not None:
            if not hasattr(self, 'q_mask_layer'):
                self.q_mask_layer = search_layer(q, q_mask)
            q_mask = self.q_mask_layer.output_mask
        if v_mask is not None:
            if not hasattr(self, 'v_mask_layer'):
                self.v_mask_layer = search_layer(v, v_mask)
            v_mask = self.v_mask_layer.output_mask
        # Pooling
        if self.pool_size > 1:
            is_self_attention = (q is k is v)
            q_in_len = K.shape(q)[1]
            q = sequence_masking(q, q_mask, 0)
            q = divisible_temporal_padding(q, self.pool_size)
            q = pool1d(q, self.pool_size, self.pool_size, pool_mode='avg')
            if is_self_attention:
                k = v = q
            else:
                k = sequence_masking(k, v_mask, 0)
                k = divisible_temporal_padding(k, self.pool_size)
                k = pool1d(k, self.pool_size, self.pool_size, pool_mode='avg')
                v = sequence_masking(v, v_mask, 0)
                v = divisible_temporal_padding(v, self.pool_size)
                v = pool1d(v, self.pool_size, self.pool_size, pool_mode='avg')
            if v_mask is not None:
                v_mask = v_mask[:, ::self.pool_size]
            if a_mask is not None and not is_string(a_mask):
                a_mask = a_mask[..., ::self.pool_size, ::self.pool_size]
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 相对位置编码
        if self.max_relative_position is not None:
            q_idxs = K.arange(0, K.shape(q)[1], dtype='int32')
            q_idxs = K.expand_dims(q_idxs, 1)
            v_idxs = K.arange(0, K.shape(v)[1], dtype='int32')
            v_idxs = K.expand_dims(v_idxs, 0)
            pos_ids = v_idxs - q_idxs
            pos_ids = K.clip(pos_ids, -self.max_relative_position,
                             self.max_relative_position)
            pos_ids = pos_ids + self.max_relative_position
            pos_embeddings = K.gather(self.relative_embeddings, pos_ids)
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, pos_embeddings)
        # Attention（续）
        a = a / self.key_size**0.5
        a = sequence_masking(a, v_mask, 1, -1)
        if a_mask is not None:
            if is_string(a_mask):
                ones = K.ones_like(a[:1, :1])
                a_mask = (ones - tf.linalg.band_part(ones, -1, 0)) * 1e12
                a = a - a_mask
            else:
                a = a - (1 - a_mask) * 1e12
        a = K.softmax(a)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', a, vw)
        if self.max_relative_position is not None:
            o = o + tf.einsum('bhjk,jkd->bjhd', a, pos_embeddings)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.o_dense(o)
        # 恢复长度
        if self.pool_size > 1:
            o = K.repeat_elements(o, self.pool_size, 1)[:, :q_in_len]
        # 返回结果
        o = sequence_masking(o, q_mask, 0)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'key_size': self.key_size,
            'pool_size': self.pool_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'max_relative_position': self.max_relative_position,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """
    def __init__(self,
                 conditional=False,
                 hidden_units=None,
                 hidden_activation='linear',
                 hidden_initializer='glorot_uniform',
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = K.epsilon() * K.epsilon()

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1], )
        else:
            shape = (input_shape[-1], )

        self.gamma = self.add_weight(shape=shape,
                                     initializer='ones',
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer='zeros',
                                    name='beta')

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer)

            self.beta_dense = Dense(units=shape[0],
                                    use_bias=False,
                                    kernel_initializer='zeros')
            self.gamma_dense = Dense(units=shape[0],
                                     use_bias=False,
                                     kernel_initializer='zeros')

    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            beta = self.beta_dense(cond)
            gamma = self.gamma_dense(cond)
            beta, gamma = self.beta + beta, self.gamma + gamma
        else:
            beta, gamma = self.beta, self.gamma

        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * gamma + beta
        return outputs

    def get_config(self):
        config = {
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer': initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionEmbedding(Layer):
    """定义位置Embedding，这里的Embedding是可训练的。
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 merge_mode='add',
                 embeddings_initializer='zeros',
                 **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.embeddings_initializer = initializers.get(embeddings_initializer)

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
        )

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embeddings = self.embeddings[:seq_len]
        pos_embeddings = K.expand_dims(pos_embeddings, 0)

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        else:
            pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim, )

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GroupDense(Layer):
    """分组全连接
    输入输出跟普通Dense一样，但参数更少，速度更快。
    """
    def __init__(self,
                 units,
                 groups=2,
                 activation='linear',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(GroupDense, self).__init__(**kwargs)
        self.units = units
        self.groups = groups
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(GroupDense, self).build(input_shape)
        input_dim = input_shape[-1]
        if not isinstance(input_dim, int):
            input_dim = input_dim.value
        assert input_dim % self.groups == 0
        assert self.units % self.groups == 0
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dim // self.groups,
                                             self.units // self.groups,
                                             self.groups),
                                      initializer=self.kernel_initializer)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units, ),
                                    initializer='zeros')

    def call(self, inputs):
        ndim, shape = K.ndim(inputs), K.shape(inputs)
        shape = [shape[i] for i in range(ndim)]
        inputs = K.reshape(inputs, shape[:-1] + [shape[-1] // self.groups, self.groups])
        outputs = tf.einsum('...ig,ijg->...gj', inputs, self.kernel)
        outputs = K.reshape(outputs, shape[:-1] + [self.units])
        outputs = outputs + self.bias
        outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units, )

    def get_config(self):
        config = {
            'units': self.units,
            'groups': self.groups,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        }
        base_config = super(GroupDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(Layer):
    """FeedForward层，其实就是两个Dense层的叠加
    """
    def __init__(self,
                 units,
                 groups=1,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 pool_size=None,
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.groups = groups
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.pool_size = pool_size or 1

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        if not isinstance(output_dim, int):
            output_dim = output_dim.value

        if self.groups is None or self.groups == 1:
            self.dense_1 = Dense(units=self.units,
                                 activation=self.activation,
                                 kernel_initializer=self.kernel_initializer)
            self.dense_2 = Dense(units=output_dim,
                                 kernel_initializer=self.kernel_initializer)
        else:
            self.dense_1 = GroupDense(units=self.units,
                                      groups=self.groups,
                                      activation=self.activation,
                                      kernel_initializer=self.kernel_initializer)
            self.dense_2 = GroupDense(units=output_dim,
                                      groups=self.groups,
                                      kernel_initializer=self.kernel_initializer)

    def call(self, inputs, mask=None):
        x = inputs
        # Pooling
        if self.pool_size > 1:
            if mask is not None:
                if not hasattr(self, 'mask_layer'):
                    self.mask_layer = search_layer(x, mask)
                mask = self.mask_layer.output_mask
            x_in_len = K.shape(x)[1]
            x = sequence_masking(x, mask, 0)
            x = divisible_temporal_padding(x, self.pool_size)
            x = pool1d(x, self.pool_size, self.pool_size, pool_mode='avg')
        # 执行FFN
        x = self.dense_1(x)
        x = self.dense_2(x)
        # 恢复长度
        if self.pool_size > 1:
            x = K.repeat_elements(x, self.pool_size, 1)[:, :x_in_len]
        # 返回结果
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'pool_size': self.pool_size,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EmbeddingDense(Layer):
    """运算跟Dense一致，但kernel用Embedding层的embeddings矩阵。
    根据Embedding层的名字来搜索定位Embedding层。
    """
    def __init__(self, embedding_name, activation='softmax', **kwargs):
        super(EmbeddingDense, self).__init__(**kwargs)
        self.embedding_name = embedding_name
        self.activation = activations.get(activation)

    def call(self, inputs):
        if not hasattr(self, 'kernel'):
            embedding_layer = search_layer(inputs, self.embedding_name)
            if embedding_layer is None:
                raise Exception('Embedding layer not found')

            self.kernel = K.transpose(embedding_layer.embeddings)
            self.units = K.int_shape(self.kernel)[1]
            self.bias = self.add_weight(name='bias',
                                        shape=(self.units, ),
                                        initializer='zeros')

        outputs = K.dot(inputs, self.kernel)
        outputs = K.bias_add(outputs, self.bias)
        outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units, )

    def get_config(self):
        config = {
            'embedding_name': self.embedding_name,
            'activation': activations.serialize(self.activation),
        }
        base_config = super(EmbeddingDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConditionalRandomField(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层。
    """
    def __init__(self, lr_multiplier=1, **kwargs):
        super(ConditionalRandomField, self).__init__(**kwargs)
        self.lr_multiplier = lr_multiplier  # 当前层学习率的放大倍数

    def build(self, input_shape):
        output_dim = input_shape[-1]
        if not isinstance(output_dim, int):
            output_dim = output_dim.value
        self.trans = self.add_weight(name='trans',
                                     shape=(output_dim, output_dim),
                                     initializer='glorot_uniform',
                                     trainable=True)
        if self.lr_multiplier != 1:
            K.set_value(self.trans, K.eval(self.trans) / self.lr_multiplier)
            self.trans = self.lr_multiplier * self.trans

    def target_score(self, y_true, y_pred, mask=None):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        """
        y_true = sequence_masking(y_true, mask, 0)
        point_score = tf.einsum('bni,bni->b', y_true, y_pred)  # 逐标签得分
        trans_score = tf.einsum('bni,ij,bnj->b', y_true[:, :-1], self.trans,
                                y_true[:, 1:])  # 标签转移得分
        return point_score + trans_score

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        outputs = tf.reduce_logsumexp(states + trans, 1)  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def call(self, inputs, mask=None):
        """CRF本身不改变输出，它只是一个loss
        """
        if mask is not None:
            if not hasattr(self, 'mask_layer'):
                self.mask_layer = search_layer(inputs, mask)

        return inputs

    @property
    def output_mask(self):
        if hasattr(self, 'mask_layer'):
            return self.mask_layer.output_mask

    def dense_loss(self, y_true, y_pred):
        """y_true需要是one hot形式
        """
        mask = self.output_mask
        # 计算目标分数
        target_score = self.target_score(y_true, y_pred, mask)
        # 递归计算log Z
        init_states = [y_pred[:, 0]]
        if mask is None:
            mask = K.ones_like(y_pred[:, :, :1])
        else:
            mask = K.expand_dims(mask, 2)
        y_pred = K.concatenate([y_pred, mask])
        log_norm, _, _ = K.rnn(self.log_norm_step,
                               y_pred[:, 1:],
                               init_states)  # 最后一步的log Z向量
        log_norm = tf.reduce_logsumexp(log_norm, 1)  # logsumexp得标量
        # 计算损失 -log p
        return log_norm - target_score

    def sparse_loss(self, y_true, y_pred):
        """y_true需要是整数形式（非one hot）
        """
        # y_true需要重新明确一下dtype和shape
        y_true = K.cast(y_true, 'int32')
        y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
        # 转为one hot
        y_true = K.one_hot(y_true, K.shape(self.trans)[0])
        return self.dense_loss(y_true, y_pred)

    def dense_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是one hot形式
        """
        y_true = K.argmax(y_true, 2)
        return self.sparse_accuracy(y_true, y_pred)

    def sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        mask = self.output_mask
        # y_true需要重新明确一下dtype和shape
        y_true = K.cast(y_true, 'int32')
        y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
        # 逐标签取最大来粗略评测训练效果
        y_pred = K.cast(K.argmax(y_pred, 2), 'int32')
        isequal = K.cast(K.equal(y_true, y_pred), K.floatx())
        if mask is None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)

    def get_config(self):
        config = {
            'lr_multiplier': self.lr_multiplier,
        }
        base_config = super(ConditionalRandomField, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaximumEntropyMarkovModel(Layer):
    """（双向）最大熵隐马尔可夫模型
    作用和用法都类似CRF，但是比CRF更快更简单。
    """
    def __init__(self, lr_multiplier=1, hidden_dim=None, **kwargs):
        super(MaximumEntropyMarkovModel, self).__init__(**kwargs)
        self.lr_multiplier = lr_multiplier  # 当前层学习率的放大倍数
        self.hidden_dim = hidden_dim  # 如果非None，则将转移矩阵低秩分解

    def build(self, input_shape):
        output_dim = input_shape[-1]
        if not isinstance(output_dim, int):
            output_dim = output_dim.value

        if self.hidden_dim is None:
            self.trans = self.add_weight(name='trans',
                                         shape=(output_dim, output_dim),
                                         initializer='glorot_uniform',
                                         trainable=True)
            if self.lr_multiplier != 1:
                K.set_value(self.trans, K.eval(self.trans) / self.lr_multiplier)
                self.trans = self.lr_multiplier * self.trans
        else:
            self.l_trans = self.add_weight(name='l_trans',
                                           shape=(output_dim, self.hidden_dim),
                                           initializer='glorot_uniform',
                                           trainable=True)
            self.r_trans = self.add_weight(name='r_trans',
                                           shape=(output_dim, self.hidden_dim),
                                           initializer='glorot_uniform',
                                           trainable=True)

            if self.lr_multiplier != 1:
                K.set_value(self.l_trans, K.eval(self.l_trans) / self.lr_multiplier)
                self.l_trans = self.lr_multiplier * self.l_trans
                K.set_value(self.r_trans, K.eval(self.r_trans) / self.lr_multiplier)
                self.r_trans = self.lr_multiplier * self.r_trans

    def call(self, inputs, mask=None):
        """MEMM本身不改变输出，它只是一个loss
        """
        if mask is not None:
            if not hasattr(self, 'mask_layer'):
                self.mask_layer = search_layer(inputs, mask)

        return inputs

    @property
    def output_mask(self):
        if hasattr(self, 'mask_layer'):
            return self.mask_layer.output_mask

    def reverse_sequence(self, inputs, mask=None):
        if mask is None:
            return [x[:, ::-1] for x in inputs]
        else:
            length = K.cast(K.sum(mask, 1), 'int32')
            return [
                tf.reverse_sequence(x, length, seq_axis=1)
                for x in inputs
            ]

    def basic_loss(self, y_true, y_pred, go_backwards=False):
        """y_true需要是整数形式（非one hot）
        """
        mask = self.output_mask
        # y_true需要重新明确一下dtype和shape
        y_true = K.cast(y_true, 'int32')
        y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
        # 反转相关
        if self.hidden_dim is None:
            if go_backwards:  # 是否反转序列
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                trans = K.transpose(self.trans)
            else:
                trans = self.trans
            histoty = K.gather(trans, y_true)
        else:
            if go_backwards:  # 是否反转序列
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                r_trans, l_trans = self.l_trans, self.r_trans
            else:
                l_trans, r_trans = self.l_trans, self.r_trans
            histoty = K.gather(l_trans, y_true)
            histoty = tf.einsum('bnd,kd->bnk', histoty, r_trans)
        # 计算loss
        histoty = K.concatenate([y_pred[:, :1], histoty[:, :-1]], 1)
        y_pred = (y_pred + histoty) / 2
        loss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        if mask is None:
            return K.mean(loss)
        else:
            return K.sum(loss * mask) / K.sum(mask)

    def sparse_loss(self, y_true, y_pred):
        """y_true需要是整数形式（非one hot）
        """
        loss = self.basic_loss(y_true, y_pred, False)
        loss = loss + self.basic_loss(y_true, y_pred, True)
        return loss / 2

    def dense_loss(self, y_true, y_pred):
        """y_true需要是one hot形式
        """
        y_true = K.argmax(y_true, 2)
        return self.sparse_loss(y_true, y_pred)

    def basic_accuracy(self, y_true, y_pred, go_backwards=False):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        mask = self.output_mask
        # y_true需要重新明确一下dtype和shape
        y_true = K.cast(y_true, 'int32')
        y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
        # 反转相关
        if self.hidden_dim is None:
            if go_backwards:  # 是否反转序列
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                trans = K.transpose(self.trans)
            else:
                trans = self.trans
            histoty = K.gather(trans, y_true)
        else:
            if go_backwards:  # 是否反转序列
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                r_trans, l_trans = self.l_trans, self.r_trans
            else:
                l_trans, r_trans = self.l_trans, self.r_trans
            histoty = K.gather(l_trans, y_true)
            histoty = tf.einsum('bnd,kd->bnk', histoty, r_trans)
        # 计算逐标签accuracy
        histoty = K.concatenate([y_pred[:, :1], histoty[:, :-1]], 1)
        y_pred = (y_pred + histoty) / 2
        y_pred = K.cast(K.argmax(y_pred, 2), 'int32')
        isequal = K.cast(K.equal(y_true, y_pred), K.floatx())
        if mask is None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)

    def sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        accuracy = self.basic_accuracy(y_true, y_pred, False)
        accuracy = accuracy + self.basic_accuracy(y_true, y_pred, True)
        return accuracy / 2

    def dense_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是one hot形式
        """
        y_true = K.argmax(y_true, 2)
        return self.sparse_accuracy(y_true, y_pred)

    def get_config(self):
        config = {
            'lr_multiplier': self.lr_multiplier,
            'hidden_dim': self.hidden_dim,
        }
        base_config = super(MaximumEntropyMarkovModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'ZeroMasking': ZeroMasking,
    'MultiHeadAttention': MultiHeadAttention,
    'LayerNormalization': LayerNormalization,
    'PositionEmbedding': PositionEmbedding,
    'GroupDense': GroupDense,
    'FeedForward': FeedForward,
    'EmbeddingDense': EmbeddingDense,
    'ConditionalRandomField': ConditionalRandomField,
    'MaximumEntropyMarkovModel': MaximumEntropyMarkovModel,
}

keras.utils.get_custom_objects().update(custom_objects)
