#! -*- coding: utf-8 -*-
# 主要模型

import numpy as np
from bert4keras.layers import *
import json
import warnings


class Transformer(object):
    """模型基类
    """
    def __init__(
            self,
            vocab_size,  # 词表大小
            hidden_size,  # 编码维度
            num_hidden_layers,  # Transformer总层数
            num_attention_heads,  # Attention的头数
            intermediate_size,  # FeedForward的隐层维度
            hidden_act,  # FeedForward隐层的激活函数
            dropout_rate,  # Dropout比例
            embedding_size=None,  # 是否指定embedding_size
            keep_tokens=None,  # 要保留的词ID列表
    ):
        if keep_tokens is None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = len(keep_tokens)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.keep_tokens = keep_tokens

    def build(self,
              layer_norm_cond=None,
              layer_norm_cond_hidden_size=None,
              layer_norm_cond_hidden_act=None,
              additional_input_layers=None):
        """模型构建函数
        layer_norm_*系列参数为实现Conditional Layer Normalization时使用，
        用来实现以“固定长度向量”为条件的条件Bert。
        """
        # Input
        self.inputs = self.prepare_inputs()
        outputs = self.inputs[:]
        if additional_input_layers is not None:
            if not isinstance(additional_input_layers, list):
                additional_input_layers = [additional_input_layers]
            self.inputs.extend(additional_input_layers)
        # Other
        layer_norm_conds = [
            layer_norm_cond,
            layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act or 'linear',
        ]
        outputs.append(layer_norm_conds)
        # Embedding
        outputs = self.prepare_embeddings(outputs)
        # Main
        for i in range(self.num_hidden_layers):
            outputs = self.prepare_main_layers(outputs)
        # Final
        outputs = self.prepare_final_layers(outputs)
        self.outputs = outputs
        # Model
        self.model = keras.models.Model(self.inputs, self.outputs)

    def prepare_inputs(self):
        raise NotImplementedError

    def prepare_embeddings(self, inputs):
        raise NotImplementedError

    def prepare_main_layers(self, inputs):
        raise NotImplementedError

    def prepare_final_layers(self, inputs):
        raise NotImplementedError

    def compute_attention_mask(self, idx):
        """定义每一层的Attention Mask
        """
        return None

    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(stddev=0.02)

    def simplify(self, inputs):
        """将list中的None过滤掉
        """
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            inputs = inputs[0]

        return inputs

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        return tf.train.load_variable(checkpoint, name)

    def create_variable(self, name, value):
        """在tensorflow中创建一个变量
        """
        return tf.Variable(value, name=name)

    def variable_mapping(self):
        """构建keras层与checkpoint的变量名之间的映射表
        """
        return {}

    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        mapping = mapping or self.variable_mapping()

        weight_value_pairs = []
        for layer, variables in mapping.items():
            layer = self.model.get_layer(layer)
            weights = layer.trainable_weights
            values = [self.load_variable(checkpoint, v) for v in variables]
            weight_value_pairs.extend(zip(weights, values))

        K.batch_set_value(weight_value_pairs)

    def save_weights_as_checkpoint(self, filename, mapping=None):
        """根据mapping将权重保存为checkpoint格式
        """
        mapping = mapping or self.variable_mapping()

        with tf.Graph().as_default():
            for layer, variables in mapping.items():
                layer = self.model.get_layer(layer)
                values = K.batch_get_value(layer.trainable_weights)
                for name, value in zip(variables, values):
                    self.create_variable(name, value)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.save(sess, filename, write_meta_graph=False)


class BERT(Transformer):
    """构建BERT模型
    """
    def __init__(
            self,
            max_position,  # 序列最大长度
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.with_pool = with_pool
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm

    def prepare_inputs(self):
        """BERT的输入是token_ids和segment_ids
        """
        x_in = Input(shape=(None, ), name='Input-Token')
        s_in = Input(shape=(None, ), name='Input-Segment')
        return [x_in, s_in]

    def prepare_embeddings(self, inputs):
        """BERT是embedding是token、position、segment三者embedding之和
        """
        x, s, layer_norm_conds = inputs
        z = layer_norm_conds[0]

        x = Embedding(input_dim=self.vocab_size,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      mask_zero=True,
                      name='Embedding-Token')(x)
        s = Embedding(input_dim=2,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      name='Embedding-Segment')(s)
        x = Add(name='Embedding-Token-Segment')([x, s])
        x = PositionEmbedding(input_dim=self.max_position,
                              output_dim=self.embedding_size,
                              merge_mode='add',
                              embeddings_initializer=self.initializer,
                              name='Embedding-Position')(x)
        x = LayerNormalization(conditional=(z is not None),
                               hidden_units=layer_norm_conds[1],
                               hidden_activation=layer_norm_conds[2],
                               hidden_initializer=self.initializer,
                               name='Embedding-Norm')(self.simplify([x, z]))
        if self.dropout_rate > 0:
            x = Dropout(rate=self.dropout_rate, name='Embedding-Dropout')(x)
        if self.embedding_size != self.hidden_size:
            x = Dense(units=self.hidden_size,
                      kernel_initializer=self.initializer,
                      name='Embedding-Mapping')(x)

        return [x, layer_norm_conds]

    def prepare_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        if not hasattr(self, 'main_count'):
            self.main_count = 0
        self.main_count += 1

        x, layer_norm_conds = inputs
        z = layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % self.main_count
        feed_forward_name = 'Transformer-%d-FeedForward' % self.main_count
        attention_mask = self.compute_attention_mask(self.main_count)

        layers = [
            MultiHeadAttention(heads=self.num_attention_heads,
                               head_size=self.attention_head_size,
                               kernel_initializer=self.initializer,
                               name=attention_name),
            Dropout(rate=self.dropout_rate,
                    name='%s-Dropout' % attention_name),
            Add(name='%s-Add' % attention_name),
            LayerNormalization(conditional=(z is not None),
                               hidden_units=layer_norm_conds[1],
                               hidden_activation=layer_norm_conds[2],
                               hidden_initializer=self.initializer,
                               name='%s-Norm' % attention_name),
            FeedForward(units=self.intermediate_size,
                        activation=self.hidden_act,
                        kernel_initializer=self.initializer,
                        name=feed_forward_name),
            Dropout(rate=self.dropout_rate,
                    name='%s-Dropout' % feed_forward_name),
            Add(name='%s-Add' % feed_forward_name),
            LayerNormalization(conditional=(z is not None),
                               hidden_units=layer_norm_conds[1],
                               hidden_activation=layer_norm_conds[2],
                               hidden_initializer=self.initializer,
                               name='%s-Norm' % feed_forward_name),
        ]

        # Self Attention
        xi, x = x, [x, x, x]
        if attention_mask is None:
            x = layers[0](x)
        elif attention_mask is 'history_only':
            x = layers[0](x, a_mask=True)
        else:
            x.append(attention_mask)
            x = layers[0](x, a_mask=True)
        if self.dropout_rate > 0:
            x = layers[1](x)
        x = layers[2]([xi, x])
        x = layers[3](self.simplify([x, z]))

        # Feed Forward
        xi = x
        x = layers[4](x)
        if self.dropout_rate > 0:
            x = layers[5](x)
        x = layers[6]([xi, x])
        x = layers[7](self.simplify([x, z]))

        return [x, layer_norm_conds]

    def prepare_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        x, layer_norm_conds = inputs
        z = layer_norm_conds[0]
        outputs = [x]

        if self.with_pool or self.with_nsp:
            # Pooler部分（提取CLS向量）
            x = outputs[0]
            x = Lambda(lambda x: x[:, 0], name='Pooler')(x)
            pool_activation = 'tanh' if self.with_pool is True else self.with_pool
            x = Dense(units=self.hidden_size,
                      activation=pool_activation,
                      kernel_initializer=self.initializer,
                      name='Pooler-Dense')(x)
            if self.with_nsp:
                # Next Sentence Prediction部分
                x = Dense(units=2,
                          activation='softmax',
                          kernel_initializer=self.initializer,
                          name='NSP-Proba')(x)
            outputs.append(x)

        if self.with_mlm:
            # Masked Language Model部分
            x = outputs[0]
            x = Dense(units=self.embedding_size,
                      activation=self.hidden_act,
                      kernel_initializer=self.initializer,
                      name='MLM-Dense')(x)
            x = LayerNormalization(conditional=(z is not None),
                                   hidden_units=layer_norm_conds[1],
                                   hidden_activation=layer_norm_conds[2],
                                   hidden_initializer=self.initializer,
                                   name='MLM-Norm')(self.simplify([x, z]))
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = EmbeddingDense(embedding_name='Embedding-Token',
                               activation=mlm_activation,
                               name='MLM-Proba')(x)
            outputs.append(x)

        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        return outputs

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(BERT, self).load_variable(checkpoint, name)
        if name in [
                'bert/embeddings/word_embeddings',
                'cls/predictions/output_bias',
        ]:
            if self.keep_tokens is None:
                return variable
            else:
                return variable[self.keep_tokens]
        elif name == 'cls/seq_relationship/output_weights':
            return variable.T
        else:
            return variable

    def create_variable(self, name, value):
        """在tensorflow中创建一个变量
        """
        if name == 'cls/seq_relationship/output_weights':
            value = value.T
        return super(BERT, self).create_variable(name, value)

    def variable_mapping(self):
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Proba': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % (i + 1): [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % (i + 1): [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % (i + 1): [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % (i + 1): [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        layer_names = set([layer.name for layer in self.model.layers])
        mapping = {k: v for k, v in mapping.items() if k in layer_names}

        return mapping


class ALBERT(BERT):
    """构建ALBERT模型
    """
    def prepare_main_layers(self, inputs):
        """ALBERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x, layer_norm_conds = inputs
        z = layer_norm_conds[0]

        if not hasattr(self, 'main_layers'):

            attention_name = 'Transformer-MultiHeadSelfAttention'
            feed_forward_name = 'Transformer-FeedForward'

            self.main_layers = [
                MultiHeadAttention(heads=self.num_attention_heads,
                                   head_size=self.attention_head_size,
                                   kernel_initializer=self.initializer,
                                   name=attention_name),
                Dropout(rate=self.dropout_rate,
                        name='%s-Dropout' % attention_name),
                Add(name='%s-Add' % attention_name),
                LayerNormalization(conditional=(z is not None),
                                   hidden_units=layer_norm_conds[1],
                                   hidden_activation=layer_norm_conds[2],
                                   hidden_initializer=self.initializer,
                                   name='%s-Norm' % attention_name),
                FeedForward(units=self.intermediate_size,
                            activation=self.hidden_act,
                            kernel_initializer=self.initializer,
                            name=feed_forward_name),
                Dropout(rate=self.dropout_rate,
                        name='%s-Dropout' % feed_forward_name),
                Add(name='%s-Add' % feed_forward_name),
                LayerNormalization(conditional=(z is not None),
                                   hidden_units=layer_norm_conds[1],
                                   hidden_activation=layer_norm_conds[2],
                                   hidden_initializer=self.initializer,
                                   name='%s-Norm' % feed_forward_name),
            ]

        # 重用层
        layers = self.main_layers
        attention_mask = self.compute_attention_mask(1)

        # Self Attention
        xi, x = x, [x, x, x]
        if attention_mask is None:
            x = layers[0](x)
        elif attention_mask is 'history_only':
            x = layers[0](x, a_mask=True)
        else:
            x.append(attention_mask)
            x = layers[0](x, a_mask=True)
        if self.dropout_rate > 0:
            x = layers[1](x)
        x = layers[2]([xi, x])
        x = layers[3](self.simplify([x, z]))

        # Feed Forward
        xi = x
        x = layers[4](x)
        if self.dropout_rate > 0:
            x = layers[5](x)
        x = layers[6]([xi, x])
        x = layers[7](self.simplify([x, z]))

        return [x, layer_norm_conds]

    def variable_mapping(self):
        """映射到官方ALBERT权重格式
        """
        mapping = super(ALBERT, self).variable_mapping()

        prefix = 'bert/encoder/transformer/group_0/inner_group_0/'
        mapping.update({
            'Transformer-MultiHeadSelfAttention': [
                prefix + 'attention_1/self/query/kernel',
                prefix + 'attention_1/self/query/bias',
                prefix + 'attention_1/self/key/kernel',
                prefix + 'attention_1/self/key/bias',
                prefix + 'attention_1/self/value/kernel',
                prefix + 'attention_1/self/value/bias',
                prefix + 'attention_1/output/dense/kernel',
                prefix + 'attention_1/output/dense/bias',
            ],
            'Transformer-MultiHeadSelfAttention-Norm': [
                prefix + 'LayerNorm/beta',
                prefix + 'LayerNorm/gamma',
            ],
            'Transformer-FeedForward': [
                prefix + 'ffn_1/intermediate/dense/kernel',
                prefix + 'ffn_1/intermediate/dense/bias',
                prefix + 'ffn_1/intermediate/output/dense/kernel',
                prefix + 'ffn_1/intermediate/output/dense/bias',
            ],
            'Transformer-FeedForward-Norm': [
                prefix + 'LayerNorm_1/beta',
                prefix + 'LayerNorm_1/gamma',
            ],
        })

        layer_names = set([layer.name for layer in self.model.layers])
        mapping = {k: v for k, v in mapping.items() if k in layer_names}

        return mapping


class ALBERT_Unshared(BERT):
    """解开ALBERT共享约束，当成BERT用
    """
    def variable_mapping(self):
        """映射到官方ALBERT权重格式
        """
        mapping = super(ALBERT_Unshared, self).variable_mapping()

        prefix = 'bert/encoder/transformer/group_0/inner_group_0/'
        for i in range(self.num_hidden_layers):
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % (i + 1): [
                    prefix + 'attention_1/self/query/kernel',
                    prefix + 'attention_1/self/query/bias',
                    prefix + 'attention_1/self/key/kernel',
                    prefix + 'attention_1/self/key/bias',
                    prefix + 'attention_1/self/value/kernel',
                    prefix + 'attention_1/self/value/bias',
                    prefix + 'attention_1/output/dense/kernel',
                    prefix + 'attention_1/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % (i + 1): [
                    prefix + 'LayerNorm/beta',
                    prefix + 'LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % (i + 1): [
                    prefix + 'ffn_1/intermediate/dense/kernel',
                    prefix + 'ffn_1/intermediate/dense/bias',
                    prefix + 'ffn_1/intermediate/output/dense/kernel',
                    prefix + 'ffn_1/intermediate/output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % (i + 1): [
                    prefix + 'LayerNorm_1/beta',
                    prefix + 'LayerNorm_1/gamma',
                ],
            })

        layer_names = set([layer.name for layer in self.model.layers])
        mapping = {k: v for k, v in mapping.items() if k in layer_names}

        return mapping


class NEZHA(BERT):
    """华为推出的NAZHA模型
    链接：https://arxiv.org/abs/1909.00204
    """
    def prepare_embeddings(self, inputs):
        """NEZHA是embedding是token、segment两者embedding之和
        """
        x, s, layer_norm_conds = inputs
        z = layer_norm_conds[0]

        x = Embedding(input_dim=self.vocab_size,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      mask_zero=True,
                      name='Embedding-Token')(x)
        s = Embedding(input_dim=2,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      name='Embedding-Segment')(s)
        x = Add(name='Embedding-Token-Segment')([x, s])
        x = LayerNormalization(conditional=(z is not None),
                               hidden_units=layer_norm_conds[1],
                               hidden_activation=layer_norm_conds[2],
                               hidden_initializer=self.initializer,
                               name='Embedding-Norm')(self.simplify([x, z]))
        if self.dropout_rate > 0:
            x = Dropout(rate=self.dropout_rate, name='Embedding-Dropout')(x)
        if self.embedding_size != self.hidden_size:
            x = Dense(units=self.hidden_size,
                      kernel_initializer=self.initializer,
                      name='Embedding-Mapping')(x)

        return [x, layer_norm_conds]

    def prepare_main_layers(self, inputs):
        """NEZHA的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        if not hasattr(self, 'main_count'):
            self.main_count = 0
        self.main_count += 1

        x, layer_norm_conds = inputs
        z = layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % self.main_count
        feed_forward_name = 'Transformer-%d-FeedForward' % self.main_count
        attention_mask = self.compute_attention_mask(self.main_count)

        layers = [
            MultiHeadAttention(heads=self.num_attention_heads,
                               head_size=self.attention_head_size,
                               kernel_initializer=self.initializer,
                               max_relative_position=64,
                               name=attention_name),
            Dropout(rate=self.dropout_rate,
                    name='%s-Dropout' % attention_name),
            Add(name='%s-Add' % attention_name),
            LayerNormalization(conditional=(z is not None),
                               hidden_units=layer_norm_conds[1],
                               hidden_activation=layer_norm_conds[2],
                               hidden_initializer=self.initializer,
                               name='%s-Norm' % attention_name),
            FeedForward(units=self.intermediate_size,
                        activation=self.hidden_act,
                        kernel_initializer=self.initializer,
                        name=feed_forward_name),
            Dropout(rate=self.dropout_rate,
                    name='%s-Dropout' % feed_forward_name),
            Add(name='%s-Add' % feed_forward_name),
            LayerNormalization(conditional=(z is not None),
                               hidden_units=layer_norm_conds[1],
                               hidden_activation=layer_norm_conds[2],
                               hidden_initializer=self.initializer,
                               name='%s-Norm' % feed_forward_name),
        ]

        # Self Attention
        xi, x = x, [x, x, x]
        if attention_mask is None:
            x = layers[0](x)
        elif attention_mask is 'history_only':
            x = layers[0](x, a_mask=True)
        else:
            x.append(attention_mask)
            x = layers[0](x, a_mask=True)
        if self.dropout_rate > 0:
            x = layers[1](x)
        x = layers[2]([xi, x])
        x = layers[3](self.simplify([x, z]))

        # Feed Forward
        xi = x
        x = layers[4](x)
        if self.dropout_rate > 0:
            x = layers[5](x)
        x = layers[6]([xi, x])
        x = layers[7](self.simplify([x, z]))

        return [x, layer_norm_conds]


def extend_with_language_model(BaseModel):
    """添加下三角的Attention Mask（语言模型用）
    """
    class LanguageModel(BaseModel):
        """带下三角Attention Mask的派生模型
        """
        def __init__(self, *args, **kwargs):
            super(LanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True
            self.attention_mask = 'history_only'

        def compute_attention_mask(self, idx):
            """重载此函数即可
            """
            return self.attention_mask

    return LanguageModel


def extend_with_unified_language_model(BaseModel):
    """添加UniLM的Attention Mask（UnifiedLanguageModel用）
    """
    class UnifiedLanguageModel(BaseModel):
        """带UniLM的Attention Mask的派生模型
        UniLM: https://arxiv.org/abs/1905.03197
        """
        def __init__(self, *args, **kwargs):
            super(UnifiedLanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True
            self.attention_mask = None

        def compute_attention_mask(self, idx):
            """重载此函数即可
            """
            if self.attention_mask is None:

                def unilm_mask(s):
                    import tensorflow as tf
                    seq_len = K.shape(s)[1]
                    with K.name_scope('attention_mask'):
                        ones = K.ones((1, 1, seq_len, seq_len))
                    a_mask = tf.linalg.band_part(ones, -1, 0)
                    s_ex12 = K.expand_dims(K.expand_dims(s, 1), 2)
                    s_ex13 = K.expand_dims(K.expand_dims(s, 1), 3)
                    a_mask = (1 - s_ex13) * (1 - s_ex12) + s_ex13 * a_mask
                    return a_mask

                self.attention_mask = Lambda(
                    unilm_mask, name='Attention-Mask')(self.inputs[1])

            return self.attention_mask

    return UnifiedLanguageModel


def build_transformer_model(config_path,
                            checkpoint_path=None,
                            with_pool=False,
                            with_nsp=False,
                            with_mlm=False,
                            model='bert',
                            application='encoder',
                            keep_tokens=None,
                            layer_norm_cond=None,
                            layer_norm_cond_hidden_size=None,
                            layer_norm_cond_hidden_act=None,
                            additional_input_layers=None,
                            return_keras_model=True):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    config = json.load(open(config_path))
    model, application = model.lower(), application.lower()

    models = {
        'bert': BERT,
        'albert': ALBERT,
        'albert_unshared': ALBERT_Unshared,
        'nezha': NEZHA,
    }
    MODEL = models[model]

    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unified_language_model(MODEL)

    transformer = MODEL(vocab_size=config['vocab_size'],
                        max_position=config.get('max_position_embeddings'),
                        hidden_size=config['hidden_size'],
                        num_hidden_layers=config['num_hidden_layers'],
                        num_attention_heads=config['num_attention_heads'],
                        intermediate_size=config['intermediate_size'],
                        hidden_act=config['hidden_act'],
                        dropout_rate=config['hidden_dropout_prob'],
                        embedding_size=config.get('embedding_size'),
                        with_pool=with_pool,
                        with_nsp=with_nsp,
                        with_mlm=with_mlm,
                        keep_tokens=keep_tokens)

    transformer.build(layer_norm_cond=layer_norm_cond,
                      layer_norm_cond_hidden_size=layer_norm_cond_hidden_size,
                      layer_norm_cond_hidden_act=layer_norm_cond_hidden_act,
                      additional_input_layers=additional_input_layers)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer.model
    else:
        return transformer


def build_bert_model(*args, **kwargs):
    warnings.warn('build_bert_model has been renamed as build_transformer_model.')
    warnings.warn('please use build_transformer_model.')
    if kwargs.get('application') == 'seq2seq':
        warnings.warn('application=\'seq2seq\' has been renamed as application=\'unilm\'')
        warnings.warn('please use application=\'unilm\'')
        kwargs['application'] = 'unilm'
    return build_transformer_model(*args, **kwargs)
