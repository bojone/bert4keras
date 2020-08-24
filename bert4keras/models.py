#! -*- coding: utf-8 -*-
# 主要模型

import numpy as np
from bert4keras.layers import *
from bert4keras.snippets import delete_arguments
from bert4keras.snippets import is_string
from keras.models import Model
import json


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
        dropout_rate=None,  # Dropout比例
        embedding_size=None,  # 是否指定embedding_size
        attention_key_size=None,  # Attention中Q,K的head_size
        sequence_length=None,  # 是否固定序列长度
        keep_tokens=None,  # 要保留的词ID列表
        compound_tokens=None,  # 扩展Embedding
        layers=None,  # 外部传入的Keras层
        prefix=None,  # 层名前缀
        name=None,  # 模型名称
        **kwargs
    ):
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_mask = None
        self.position_bias = None
        self.layers = {} if layers is None else layers
        self.prefix = prefix or ''
        self.name = name
        self.built = False

    def build(
        self,
        layer_norm_cond=None,
        layer_norm_cond_hidden_size=None,
        layer_norm_cond_hidden_act=None,
        additional_input_layers=None,
        **kwargs
    ):
        """模型构建函数
        layer_norm_*系列参数为实现Conditional Layer Normalization时使用，
        用来实现以“固定长度向量”为条件的条件Bert。
        """
        if self.built:
            return None
        # Input
        inputs = self.get_inputs()
        self.set_inputs(inputs, additional_input_layers)
        # Other
        self.layer_norm_conds = [
            layer_norm_cond,
            layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act or 'linear',
        ]
        # Call
        outputs = self.call(inputs)
        self.set_outputs(outputs)
        # Model
        self.model = Model(self.inputs, self.outputs, name=self.name)
        self.built = True

    def call(self, inputs):
        """定义模型的执行流程
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs

    def prefixed(self, name):
        """给名字加前缀
        """
        if name is not None:
            return self.prefix + name

    def apply(self, inputs=None, layer=None, arguments=None, **kwargs):
        """通过apply调用层会自动重用同名层
        inputs: 上一层的输出；
        layer: 要调用的层类名；
        arguments: 传递给layer.call的参数；
        kwargs: 传递给层初始化的参数。
        """
        if layer is Dropout and self.dropout_rate == 0:
            return inputs

        arguments = arguments or {}
        name = self.prefixed(kwargs.get('name'))
        kwargs['name'] = name
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name
            self.layers[name] = layer

        if inputs is None:
            return self.layers[name]
        else:
            return self.layers[name](inputs, **arguments)

    def get_inputs(self):
        raise NotImplementedError

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs, index):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError

    def compute_attention_mask(self, inputs=None):
        """定义每一层的Attention Mask
        """
        return self.attention_mask

    def compute_position_bias(self, inputs=None):
        """定义每一层的Position Bias（一般相对位置编码用）
        """
        return self.position_bias

    def set_inputs(self, inputs, additional_input_layers=None):
        """设置input和inputs属性
        """
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        inputs = inputs[:]
        if additional_input_layers is not None:
            if not isinstance(additional_input_layers, list):
                additional_input_layers = [additional_input_layers]
            inputs.extend(additional_input_layers)

        self.inputs = inputs
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]

    def set_outputs(self, outputs):
        """设置output和oututs属性
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

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

    def load_embeddings(self, embeddings):
        """处理Embedding层权重
        """
        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = np.array([
                embeddings[idxs].mean(0) for idxs in self.compound_tokens
            ])
            embeddings = np.concatenate([embeddings, ext_embeddings], 0)

        return embeddings

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        return tf.train.load_variable(checkpoint, name)

    def create_variable(self, name, value):
        """创建一个变量
        """
        return K.variable(self.initializer(value.shape), name=name)

    def variable_mapping(self):
        """构建keras层与checkpoint的变量名之间的映射表
        """
        return {}

    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        weight_value_pairs = []
        for layer, variables in mapping.items():
            layer = self.layers[layer]
            weights = layer.trainable_weights
            values = [self.load_variable(checkpoint, v) for v in variables]

            if isinstance(layer, MultiHeadAttention):
                """如果key_size不等于head_size，则可以通过
                正交矩阵将相应的权重投影到合适的shape。
                """
                count = 2
                if layer.use_bias:
                    count += 2
                heads = self.num_attention_heads
                head_size = self.attention_head_size
                key_size = self.attention_key_size
                W = np.linalg.qr(np.random.randn(key_size, head_size))[0].T
                if layer.attention_scale:
                    W = W * key_size**0.25 / head_size**0.25
                for i in range(count):
                    w, v = weights[i], values[i]
                    w_shape, v_shape = K.int_shape(w), v.shape
                    if w_shape[-1] != v_shape[-1]:
                        pre_shape = w_shape[:-1]
                        v = v.reshape(pre_shape + (heads, head_size))
                        v = np.dot(v, W)
                        v = v.reshape(pre_shape + (heads * key_size,))
                        values[i] = v

            weight_value_pairs.extend(zip(weights, values))

        K.batch_set_value(weight_value_pairs)

    def save_weights_as_checkpoint(self, filename, mapping=None):
        """根据mapping将权重保存为checkpoint格式
        """
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        with tf.Graph().as_default():
            all_variables, all_values = [], []
            for layer, variables in mapping.items():
                layer = self.layers[layer]
                values = K.batch_get_value(layer.trainable_weights)
                for name, value in zip(variables, values):
                    all_variables.append(self.create_variable(name, value))
                    all_values.append(value)
            with tf.Session() as sess:
                K.batch_set_value(zip(all_variables, all_values))
                saver = tf.train.Saver()
                saver.save(sess, filename)


class BERT(Transformer):
    """构建BERT模型
    """
    def __init__(
        self,
        max_position,  # 序列最大长度
        segment_vocab_size=2,  # segment总数目
        with_pool=False,  # 是否包含Pool部分
        with_nsp=False,  # 是否包含NSP部分
        with_mlm=False,  # 是否包含MLM部分
        custom_position_ids=False,  # 是否自行传入位置id
        **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.custom_position_ids = custom_position_ids
        if self.with_nsp and not self.with_pool:
            self.with_pool = True

    def get_inputs(self):
        """BERT的输入是token_ids和segment_ids
        （但允许自行传入位置id，以实现一些特殊需求）
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
        )
        inputs = [x_in]

        if self.segment_vocab_size > 0:
            s_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Segment'
            )
            inputs.append(s_in)

        if self.custom_position_ids:
            p_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Position'
            )
            inputs.append(p_in)

        return inputs

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
        if self.custom_position_ids:
            p = inputs.pop(0)
        else:
            p = None
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        if self.segment_vocab_size > 0:
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name='Embedding-Segment'
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )
        x = self.apply(
            inputs=self.simplify([x, p]),
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position'
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """BERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_mask(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_mask': None}
        if attention_mask is not None:
            arguments['a_mask'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        x = inputs
        z = self.layer_norm_conds[0]
        outputs = [x]

        if self.with_pool:
            # Pooler部分（提取CLS向量）
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Lambda,
                function=lambda x: x[:, 0],
                name='Pooler'
            )
            pool_activation = 'tanh' if self.with_pool is True else self.with_pool
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                activation=pool_activation,
                kernel_initializer=self.initializer,
                name='Pooler-Dense'
            )
            if self.with_nsp:
                # Next Sentence Prediction部分
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=2,
                    activation='softmax',
                    kernel_initializer=self.initializer,
                    name='NSP-Proba'
                )
            outputs.append(x)

        if self.with_mlm:
            # Masked Language Model部分
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.embedding_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name='MLM-Dense'
            )
            x = self.apply(
                inputs=self.simplify([x, z]),
                layer=LayerNormalization,
                conditional=(z is not None),
                hidden_units=self.layer_norm_conds[1],
                hidden_activation=self.layer_norm_conds[2],
                hidden_initializer=self.initializer,
                name='MLM-Norm'
            )
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={'mode': 'dense'},
                name='Embedding-Token'
            )
            x = self.apply(inputs=x, layer=BiasAdd, name='MLM-Bias')
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation=mlm_activation,
                name='MLM-Activation'
            )
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
            return self.load_embeddings(variable)
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
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        return mapping


class ALBERT(BERT):
    """构建ALBERT模型
    """
    def apply_main_layers(self, inputs, index):
        """ALBERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-MultiHeadSelfAttention'
        feed_forward_name = 'Transformer-FeedForward'
        attention_mask = self.compute_attention_mask(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_mask': None}
        if attention_mask is not None:
            arguments['a_mask'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

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
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention_1/self/query/kernel',
                    prefix + 'attention_1/self/query/bias',
                    prefix + 'attention_1/self/key/kernel',
                    prefix + 'attention_1/self/key/bias',
                    prefix + 'attention_1/self/value/kernel',
                    prefix + 'attention_1/self/value/bias',
                    prefix + 'attention_1/output/dense/kernel',
                    prefix + 'attention_1/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'LayerNorm/beta',
                    prefix + 'LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'ffn_1/intermediate/dense/kernel',
                    prefix + 'ffn_1/intermediate/dense/bias',
                    prefix + 'ffn_1/intermediate/output/dense/kernel',
                    prefix + 'ffn_1/intermediate/output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'LayerNorm_1/beta',
                    prefix + 'LayerNorm_1/gamma',
                ],
            })

        return mapping


class NEZHA(BERT):
    """华为推出的NAZHA模型
    链接：https://arxiv.org/abs/1909.00204
    """
    def apply_embeddings(self, inputs):
        """NEZHA的embedding是token、segment两者embedding之和
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        if self.segment_vocab_size > 0:
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=2,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name='Embedding-Segment'
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """NEZHA的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_mask(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi, x = x, [x, x, x, position_bias]
        arguments = {'a_mask': None, 'p_bias': 'typical_relative'}
        if attention_mask is not None:
            arguments['a_mask'] = True
            x.insert(3, attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def compute_position_bias(self, inputs=None):
        """经典相对位置编码
        """
        if self.position_bias is None:

            def sinusoidal(shape, dtype=None):
                """NEZHA直接使用Sin-Cos形式的位置向量
                """
                vocab_size, depth = shape
                embeddings = np.zeros(shape)
                for pos in range(vocab_size):
                    for i in range(depth // 2):
                        theta = pos / np.power(10000, 2. * i / depth)
                        embeddings[pos, 2 * i] = np.sin(theta)
                        embeddings[pos, 2 * i + 1] = np.cos(theta)
                return embeddings

            x = inputs
            self.position_bias = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbedding,
                input_dim=2 * 64 + 1,
                output_dim=self.attention_head_size,
                embeddings_initializer=sinusoidal,
                name='Embedding-Relative-Position',
                trainable=False
            )

        return self.position_bias


class ELECTRA(BERT):
    """Google推出的ELECTRA模型
    链接：https://arxiv.org/abs/2003.10555
    """
    @delete_arguments('with_pool', 'with_mlm')
    def __init__(
        self,
        max_position,  # 序列最大长度
        **kwargs  # 其余参数
    ):
        if 'keep_tokens' in kwargs:
            del kwargs['keep_tokens']

        super(ELECTRA, self).__init__(max_position, **kwargs)

    def apply_final_layers(self, inputs):
        x = inputs
        z = self.layer_norm_conds[0]
        return x

    def variable_mapping(self):
        mapping = super(ELECTRA, self).variable_mapping()
        mapping['Embedding-Mapping'] = [
            'electra/embeddings_project/kernel',
            'electra/embeddings_project/bias',
        ]
        mapping = {
            k: [i.replace('bert/', 'electra/') for i in v]
            for k, v in mapping.items()
        }
        return mapping


class GPT2_ML(Transformer):
    """构建GPT2_ML模型
    链接: https://github.com/imcaspar/gpt2-ml
    """
    def __init__(
        self,
        max_position,  # 序列最大长度
        final_activation='softmax',  # 预测分布的激活函数
        **kwargs  # 其余参数
    ):
        super(GPT2_ML, self).__init__(**kwargs)
        self.max_position = max_position
        self.final_activation = final_activation

    def get_inputs(self):
        """GPT2_ML的输入是token_ids和segment_ids
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
        )
        return x_in

    def apply_embeddings(self, inputs):
        """GPT2_ML的embedding是token、position两者embedding之和
        """
        x = inputs
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            embeddings_initializer=self.initializer,
            name='Embedding-Position'
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """GPT2_ML的主体是基于Self-Attention的模块
        顺序：Att  --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_mask(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x, attention_mask], {'a_mask': True}

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm-0' % feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm-1' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs
        z = self.layer_norm_conds[0]

        # Language Model部分
        x = self.apply(
            inputs=x,
            layer=Embedding,
            arguments={'mode': 'dense'},
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Activation,
            activation=self.final_activation,
            name='LM-Activation'
        )

        return x

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(GPT2_ML, self).load_variable(checkpoint, name)
        if name == 'newslm/embeddings/word_embed':
            return self.load_embeddings(variable)
        else:
            return variable

    def compute_attention_mask(self, inputs=None):
        """添加下三角形式的attention mask
        """
        if self.attention_mask is None:

            def lm_mask(s):
                seq_len = K.shape(s)[1]
                idxs = K.arange(0, seq_len)
                mask = idxs[None, :] <= idxs[:, None]
                mask = K.cast(mask, K.floatx())
                return mask[None, None]

            self.attention_mask = self.apply(
                inputs=self.inputs[0],
                layer=Lambda,
                function=lm_mask,
                name='Attention-LM-Mask'
            )

        return self.attention_mask

    def variable_mapping(self):
        """映射到官方GPT2_ML权重格式
        """
        mapping = {
            'Embedding-Token': ['newslm/embeddings/word_embed'],
            'Embedding-Position': ['newslm/embeddings/pos_embed'],
            'Embedding-Norm': [
                'newslm/embeddings/LayerNorm_embed_norm/beta',
                'newslm/embeddings/LayerNorm_embed_norm/gamma',
            ],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'newslm/layer%02d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'query_layer/kernel',
                    prefix + 'query_layer/bias',
                    prefix + 'key_layer/kernel',
                    prefix + 'key_layer/bias',
                    prefix + 'value_layer/kernel',
                    prefix + 'value_layer/bias',
                    prefix + 'context_projection_layer/kernel',
                    prefix + 'context_projection_layer/bias',
                ],
                'Transformer-%d-FeedForward-Norm-0' % i: [
                    prefix + 'LayerNorm_mlp_ln0/beta',
                    prefix + 'LayerNorm_mlp_ln0/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/kernel',
                    prefix + 'intermediate/bias',
                    prefix + 'output/kernel',
                    prefix + 'output/bias',
                ],
                'Transformer-%d-FeedForward-Norm-1' % i: [
                    prefix + 'LayerNorm_mlp_ln1/beta',
                    prefix + 'LayerNorm_mlp_ln1/gamma',
                ],
            })

        return mapping


class T5_Base(Transformer):
    """Google的T5模型（基类）
    """
    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(T5_Base, self).load_variable(checkpoint, name)
        if name == 'shared/embedding':
            return self.load_embeddings(variable)
        elif 'relative_attention_bias' in name:
            return variable.T
        else:
            return variable

    def create_variable(self, name, value):
        """在tensorflow中创建一个变量
        """
        if 'relative_attention_bias' in name:
            value = value.T
        return super(T5_Base, self).create_variable(name, value)

    def variable_mapping(self):
        """映射到官方T5权重格式
        """
        mapping = {
            'Embedding-Token': ['shared/embedding'],
            'Encoder-Embedding-Relative-Position': [
                'encoder/block_000/layer_000/SelfAttention/relative_attention_bias'
            ],
            'Encoder-Output-Norm': ['encoder/final_layer_norm/scale'],
            'Decoder-Embedding-Relative-Position': [
                'decoder/block_000/layer_000/SelfAttention/relative_attention_bias',
            ],
            'Decoder-Output-Norm': ['decoder/final_layer_norm/scale'],
        }

        for i in range(self.num_hidden_layers):
            # Encoder主体
            prefix = 'encoder/block_%03d/' % i
            mapping.update({
                'Encoder-Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'layer_000/SelfAttention/q',
                    prefix + 'layer_000/SelfAttention/k',
                    prefix + 'layer_000/SelfAttention/v',
                    prefix + 'layer_000/SelfAttention/o',
                ],
                'Encoder-Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'layer_000/layer_norm/scale',
                ],
                'Encoder-Transformer-%d-FeedForward' % i: [
                    prefix + 'layer_001/DenseReluDense/wi/kernel',
                    prefix + 'layer_001/DenseReluDense/wo/kernel',
                ],
                'Encoder-Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'layer_001/layer_norm/scale',
                ],
            })
            # Decoder主体
            prefix = 'decoder/block_%03d/' % i
            mapping.update({
                'Decoder-Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'layer_000/SelfAttention/q',
                    prefix + 'layer_000/SelfAttention/k',
                    prefix + 'layer_000/SelfAttention/v',
                    prefix + 'layer_000/SelfAttention/o',
                ],
                'Decoder-Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'layer_000/layer_norm/scale',
                ],
                'Decoder-Transformer-%d-MultiHeadCrossAttention' % i: [
                    prefix + 'layer_001/EncDecAttention/q',
                    prefix + 'layer_001/EncDecAttention/k',
                    prefix + 'layer_001/EncDecAttention/v',
                    prefix + 'layer_001/EncDecAttention/o',
                ],
                'Decoder-Transformer-%d-MultiHeadCrossAttention-Norm' % i: [
                    prefix + 'layer_001/layer_norm/scale',
                ],
                'Decoder-Transformer-%d-FeedForward' % i: [
                    prefix + 'layer_002/DenseReluDense/wi/kernel',
                    prefix + 'layer_002/DenseReluDense/wo/kernel',
                ],
                'Decoder-Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'layer_002/layer_norm/scale',
                ],
            })

        return mapping


class T5_Encoder(T5_Base):
    """Google的T5模型（Encoder）
    """
    def get_inputs(self):
        """T5的Encoder的输入只有token_ids
        """
        x_in = self.apply(
            layer=Input,
            shape=(self.sequence_length,),
            name='Encoder-Input-Token'
        )
        return x_in

    def apply_embeddings(self, inputs):
        """T5的embedding只有token embedding，
        并把relative position embedding准备好，待attention使用。
        """
        x = inputs

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Encoder-Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Encoder-Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """T5的Encoder的主体是基于Self-Attention的模块
        顺序：LN --> Att --> Add --> LN --> FFN --> Add
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Encoder-Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Encoder-Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_mask(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )
        x = self.apply(
            inputs=[x, x, x, position_bias],
            layer=MultiHeadAttention,
            arguments={'p_bias': 't5_relative'},
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=False,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Encoder-Output-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Encoder-Output-Dropout'
        )

        return x

    def compute_position_bias(self, inputs=None):
        """T5相对位置编码
        """
        if self.position_bias is None:

            x = inputs
            p = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=True,
                embeddings_initializer=self.initializer,
                name='Encoder-Embedding-Relative-Position'
            )
            self.position_bias = p

        return self.position_bias


class T5_Decoder(T5_Base):
    """Google的T5模型（Decoder）
    """
    def __init__(self, with_lm=True, **kwargs):
        super(T5_Decoder, self).__init__(**kwargs)
        self.with_lm = with_lm

    def get_inputs(self):
        """T5的Decoder的输入为context序列和token_ids
        """
        c_in = self.apply(
            layer=Input,
            shape=(self.sequence_length, self.hidden_size),
            name='Input-Context'
        )
        x_in = self.apply(
            layer=Input,
            shape=(self.sequence_length,),
            name='Decoder-Input-Token'
        )
        return [c_in, x_in]

    def apply_embeddings(self, inputs):
        """T5的embedding只有token embedding，
        并把relative position embedding准备好，待attention使用。
        """
        c, x = inputs

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Decoder-Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Decoder-Embedding-Mapping'
            )

        return [c, x]

    def apply_main_layers(self, inputs, index):
        """T5的Dencoder主体是基于Self-Attention、Cross-Attention的模块
        顺序：LN --> Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add
        """
        c, x = inputs
        z = self.layer_norm_conds[0]

        self_attention_name = 'Decoder-Transformer-%d-MultiHeadSelfAttention' % index
        cross_attention_name = 'Decoder-Transformer-%d-MultiHeadCrossAttention' % index
        feed_forward_name = 'Decoder-Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_mask(index)
        position_bias = self.compute_position_bias([x, c])

        # Self Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % self_attention_name
        )
        x = self.apply(
            inputs=[x, x, x, attention_mask, position_bias[0]],
            layer=MultiHeadAttention,
            arguments={
                'a_mask': True,
                'p_bias': 't5_relative'
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            kernel_initializer=self.initializer,
            name=self_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % self_attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % self_attention_name
        )

        # Cross Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % cross_attention_name
        )
        x = self.apply(
            inputs=[x, c, c, position_bias[1]],
            layer=MultiHeadAttention,
            arguments={
                'a_mask': None,
                'p_bias': 't5_relative'
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            kernel_initializer=self.initializer,
            name=cross_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % cross_attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % cross_attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=False,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )

        return [c, x]

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        c, x = inputs
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Decoder-Output-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Decoder-Output-Dropout'
        )
        x = self.apply(
            inputs=x,
            layer=Lambda,
            function=lambda x: x / np.sqrt(self.hidden_size),
            name='Decoder-Output-Scale'
        )

        if self.with_lm:
            # 预测token概率部分
            if self.embedding_size != self.hidden_size:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.embedding_size,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-Mapping'
                )
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={'mode': 'dense'},
                name='Embedding-Token'
            )
            lm_activation = 'softmax' if self.with_lm is True else self.with_lm
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation=lm_activation,
                name='Dencoder-Output-LM-Activation'
            )

        return x

    def compute_attention_mask(self, inputs=None):
        """添加下三角形式的attention mask
        """
        if self.attention_mask is None:

            def lm_mask(s):
                seq_len = K.shape(s)[1]
                idxs = K.arange(0, seq_len)
                mask = idxs[None, :] <= idxs[:, None]
                mask = K.cast(mask, K.floatx())
                return mask[None, None]

            self.attention_mask = self.apply(
                inputs=self.inputs[1],
                layer=Lambda,
                function=lm_mask,
                name='Attention-LM-Mask'
            )

        return self.attention_mask

    def compute_position_bias(self, inputs=None):
        """T5相对位置编码
        """
        if self.position_bias is None:

            x, c = inputs
            p1 = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position'
            )
            p2 = self.apply(
                inputs=[x, c],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position'
            )
            self.position_bias = (p1, p2)

        return self.position_bias


class T5(T5_Base):
    """Google的T5模型（Encoder-Decoder）
    """
    def __init__(self, **kwargs):
        super(T5, self).__init__(**kwargs)
        kwargs['layers'] = self.layers
        e_name, d_name = 'Encoder', 'Decoder'
        if 'name' in kwargs:
            e_name = '%s_%s' % (kwargs['name'], e_name)
            d_name = '%s_%s' % (kwargs['name'], d_name)
            del kwargs['name']  # 防止重复传参
        self._encoder = T5_Encoder(name=e_name, **kwargs)
        self._decoder = T5_Decoder(name=d_name, **kwargs)

    def build(self, **kwargs):
        """同时构建Encoder和Decoder
        """
        self._encoder.build(**kwargs)
        self._decoder.build(**kwargs)
        self.encoder = self._encoder.model
        self.decoder = self._decoder.model
        self.inputs = self.encoder.inputs + self.decoder.inputs[1:]
        self.outputs = self.decoder(
            self.encoder.outputs + self.decoder.inputs[1:]
        )
        self.model = Model(self.inputs, self.outputs)


def extend_with_language_model(BaseModel):
    """添加下三角的Attention Mask（语言模型用）
    """
    class LanguageModel(BaseModel):
        """带下三角Attention Mask的派生模型
        """
        def __init__(self, *args, **kwargs):
            super(LanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True

        def compute_attention_mask(self, inputs=None):
            """重载此函数即可
            """
            if self.attention_mask is None:

                def lm_mask(s):
                    seq_len = K.shape(s)[1]
                    idxs = K.arange(0, seq_len)
                    mask = idxs[None, :] <= idxs[:, None]
                    mask = K.cast(mask, K.floatx())
                    return mask[None, None]

                self.attention_mask = self.apply(
                    inputs=self.inputs[0],
                    layer=Lambda,
                    function=lm_mask,
                    name='Attention-LM-Mask'
                )

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

        def compute_attention_mask(self, inputs=None):
            """重载此函数即可
            """
            if self.attention_mask is None:

                def unilm_mask(s):
                    idxs = K.cumsum(s, axis=1)
                    mask = idxs[:, None, :] <= idxs[:, :, None]
                    mask = K.cast(mask, K.floatx())
                    return mask[:, None]

                self.attention_mask = self.apply(
                    inputs=self.inputs[1],
                    layer=Lambda,
                    function=unilm_mask,
                    name='Attention-UniLM-Mask'
                )

            return self.attention_mask

    return UnifiedLanguageModel


def extend_with_parallel_unified_language_model(BaseModel):
    """添加并行式UniLM的Attention Mask（Seq2Seq用）
    注：用于加速一对多的Seq2Seq任务的训练过程。
    """
    class ParallelUnifiedLanguageModel(BaseModel):
        """带并行式UniLM的Attention Mask的派生模型
        UniLM: https://arxiv.org/abs/1905.03197
        """
        def __init__(self, *args, **kwargs):
            super(ParallelUnifiedLanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True
            self.custom_position_ids = True

        def compute_attention_mask(self, inputs=None):
            """重载此函数即可
            """
            if self.attention_mask is None:

                def punilm_mask(inputs):
                    s, p = inputs

                    s_idxs = K.cumsum(s, axis=1)
                    s_mask = s_idxs[:, None, :] <= s_idxs[:, :, None]
                    s_mask = K.cast(s_mask, K.floatx())

                    p = K.less(p[:, 1:] - p[:, :-1], 0)
                    p = K.cast(p, K.floatx())
                    p = K.concatenate([K.zeros_like(p[:, :1]), p], axis=1)
                    p_idxs = K.cumsum(p, axis=1) + s
                    p_mask = K.equal(p_idxs[:, None, :], p_idxs[:, :, None])
                    p_mask = K.cast(p_mask, K.floatx())

                    s = s[:, None]
                    mask = s_mask * (1 - s) + p_mask * s_mask * s
                    return mask[:, None]

                self.attention_mask = self.apply(
                    inputs=self.inputs[1:3],
                    layer=Lambda,
                    function=punilm_mask,
                    name='Attention-PUniLM-Mask'
                )

            return self.attention_mask

    return ParallelUnifiedLanguageModel


def build_transformer_model(
    config_path=None,
    checkpoint_path=None,
    model='bert',
    application='encoder',
    return_keras_model=True,
    **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings')
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size')

    models = {
        'bert': BERT,
        'albert': ALBERT,
        'albert_unshared': ALBERT_Unshared,
        'roberta': BERT,
        'nezha': NEZHA,
        'electra': ELECTRA,
        'gpt2_ml': GPT2_ML,
        't5': T5,
        't5_encoder': T5_Encoder,
        't5_decoder': T5_Decoder,
    }

    if is_string(model):
        model = model.lower()
        MODEL = models[model]
    else:
        MODEL = model

    application = application.lower()
    if application in ['lm', 'unilm', 'punilm'] and model in ['electra', 't5']:
        raise ValueError(
            '"%s" model can not be used as "%s" application.\n' %
            (model, application)
        )

    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unified_language_model(MODEL)
    elif application == 'punilm':
        MODEL = extend_with_parallel_unified_language_model(MODEL)

    transformer = MODEL(**configs)
    transformer.build(**configs)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer.model
    else:
        return transformer
