#! -*- coding: utf-8 -*-
# 自定义层

import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.snippets import get_all_attributes, is_string


# 等价于 from keras.layers import *
locals().update(get_all_attributes(keras.layers))
initializers = keras.initializers
activations = keras.activations


def sequence_masking(x, mask, mode=0, axis=None):
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


class MultiHeadAttention(Layer):
    """多头注意力机制
    """
    def __init__(self,
                 heads,
                 head_size,
                 key_size=None,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = heads * head_size
        self.key_size = key_size if key_size else head_size
        self.kernel_initializer = initializers.get(kernel_initializer)

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

    def call(self, inputs, q_mask=False, v_mask=False, a_mask=False):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        a_mask: 对attention矩阵的mask。
                不同的attention mask对应不同的应用。
        """
        q, k, v = inputs[:3]
        # 处理mask
        idx = 3
        if q_mask:
            q_mask = inputs[idx]
            idx += 1
        else:
            q_mask = None
        if v_mask:
            v_mask = inputs[idx]
            idx += 1
        else:
            v_mask = None
        if a_mask:
            if len(inputs) > idx:
                a_mask = inputs[idx]
            else:
                a_mask = 'history_only'
        else:
            a_mask = None
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw) / self.key_size**0.5
        a = sequence_masking(a, v_mask, 1, -1)
        if a_mask is not None:
            if is_string(a_mask) and a_mask == 'history_only':
                ones = K.ones_like(a[:1, :1])
                a_mask = (ones - tf.linalg.band_part(ones, -1, 0)) * 1e12
                a = a - a_mask
            else:
                a = a - (1 - a_mask) * 1e12
        a = K.softmax(a)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', a, vw)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.o_dense(o)
        o = sequence_masking(o, q_mask, 0)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'key_size': self.key_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.groups = groups
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

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

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
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
        if isinstance(activation, bool):
            activation = 'softmax'
        self.activation = activations.get(activation)

    def call(self, inputs):
        if not hasattr(self, 'kernel'):
            embedding_layer = inputs._keras_history[0]

            if embedding_layer.name != self.embedding_name:

                def recursive_search(layer):
                    """递归向上搜索，根据名字找Embedding层
                    """
                    last_layer = layer._inbound_nodes[0].inbound_layers
                    if isinstance(last_layer, list):
                        if len(last_layer) == 0:
                            return None
                        else:
                            last_layer = last_layer[0]
                    if last_layer.name == self.embedding_name:
                        return last_layer
                    else:
                        return recursive_search(last_layer)

                embedding_layer = recursive_search(embedding_layer)
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


custom_objects = {
    'MultiHeadAttention': MultiHeadAttention,
    'LayerNormalization': LayerNormalization,
    'PositionEmbedding': PositionEmbedding,
    'GroupDense': GroupDense,
    'FeedForward': FeedForward,
    'EmbeddingDense': EmbeddingDense,
}

keras.utils.get_custom_objects().update(custom_objects)
