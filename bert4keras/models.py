#! -*- coding: utf-8 -*-
# 主要模型

import numpy as np
from bert4keras.backend import sequence_masking
from bert4keras.layers import *
from bert4keras.snippets import insert_arguments
from bert4keras.snippets import delete_arguments
from bert4keras.snippets import is_string
from bert4keras.snippets import orthogonally_resize
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
        attention_head_size=None,  # Attention中V的head_size
        attention_key_size=None,  # Attention中Q,K的head_size
        sequence_length=None,  # 是否固定序列长度
        keep_tokens=None,  # 要保留的词ID列表
        compound_tokens=None,  # 扩展Embedding
        residual_attention_scores=False,  # Attention矩阵加残差
        ignore_invalid_weights=False,  # 允许跳过不存在的权重
        autoresize_weights=False,  # 自动变换形状不匹配的权重
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
        self.attention_head_size = attention_head_size or hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights
        self.autoresize_weights = autoresize_weights
        self.layers = {} if layers is None else layers
        self.prefix = prefix or ''
        self.name = name
        self.built = False

    def build(
        self,
        attention_caches=None,
        layer_norm_cond=None,
        layer_norm_cond_hidden_size=None,
        layer_norm_cond_hidden_act=None,
        additional_input_layers=None,
        **kwargs
    ):
        """模型构建函数
        attention_caches：为Attention的K,V的缓存序列字典，格式为
                         {Attention层名: [K缓存, V缓存]}；
        layer_norm_*系列参数：实现Conditional Layer Normalization时使用，
                            用来实现以“固定长度向量”为条件的条件Bert。
        """
        if self.built:
            return None
        # Input
        inputs = self.get_inputs()
        self.set_inputs(inputs, additional_input_layers)
        # Other
        self.attention_caches = attention_caches or {}
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

        if layer is MultiHeadAttention and self.residual_attention_scores:
            kwargs['return_attention_scores'] = True

        arguments = arguments or {}
        if layer is Lambda:
            kwargs['arguments'] = arguments
            arguments = {}

        name = self.prefixed(kwargs.get('name'))
        kwargs['name'] = name
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name
            self.layers[name] = layer

        if inputs is None:
            return self.layers[name]
        else:
            if isinstance(self.layers[name], MultiHeadAttention):
                if name in self.attention_caches:
                    # 如果检测到Cache的传入，那么自动在Key,Value处拼接起来
                    k_cache, v_cache = self.attention_caches[name]
                    k_name, v_name = name + '-Cached-Key', name + '-Cached-Value'
                    k = Concatenate1D(name=k_name)([k_cache, inputs[1]])
                    v = Concatenate1D(name=v_name)([v_cache, inputs[2]])
                    inputs = inputs[:1] + [k, v] + inputs[3:]
                if self.residual_attention_scores:
                    # 如果使用残差Attention矩阵，则给每个Attention矩阵加上前上一层的Attention
                    # 矩阵，这对应RealFormer设计（https://arxiv.org/abs/2012.11747）。目前
                    # 该实现还相对粗糙，可能欠缺通用性。
                    if self.attention_scores is not None:
                        if arguments.get('a_bias'):
                            a_bias = Add(name=name + '-Attention-Bias'
                                        )([inputs[3], self.attention_scores])
                            inputs = inputs[:3] + [a_bias] + inputs[4:]
                        else:
                            a_bias = self.attention_scores
                            inputs = inputs[:3] + [a_bias] + inputs[3:]
                        arguments['a_bias'] = True
                    o, a = self.layers[name](inputs, **arguments)
                    self.attention_scores = a
                    return o
            return self.layers[name](inputs, **arguments)

    def get_inputs(self):
        raise NotImplementedError

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs, index):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError

    def compute_attention_bias(self, inputs=None):
        """定义每一层的Attention Bias
        """
        return self.attention_bias

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
        embeddings = embeddings.astype(K.floatx())  # 防止np.average报错

        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                if isinstance(item, list):
                    item = (item, [1] * len(item))
                ext_embeddings.append(
                    np.average(embeddings[item[0]], 0, item[1])
                )
            embeddings = np.concatenate([embeddings, ext_embeddings], 0)

        return embeddings

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        if isinstance(checkpoint, dict):
            return checkpoint[name]
        else:
            return tf.train.load_variable(checkpoint, name)

    def create_variable(self, name, value, dtype=None):
        """创建一个变量
        """
        dtype = dtype or K.floatx()
        return K.variable(
            self.initializer(value.shape, dtype), dtype, name=name
        ), value

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
            weights, values = [], []

            for w, v in zip(layer.trainable_weights, variables):  # 允许跳过不存在的权重
                try:
                    values.append(self.load_variable(checkpoint, v))
                    weights.append(w)
                except Exception as e:
                    if self.ignore_invalid_weights:
                        print('%s, but ignored.' % e.message)
                    else:
                        raise e

            for i, (w, v) in enumerate(zip(weights, values)):
                if v is not None:
                    w_shape, v_shape = K.int_shape(w), v.shape
                    if self.autoresize_weights and w_shape != v_shape:
                        v = orthogonally_resize(v, w_shape)
                        if isinstance(layer, MultiHeadAttention):
                            count = 2
                            if layer.use_bias:
                                count += 2
                            if layer.attention_scale and i < count:
                                scale = 1.0 * w_shape[-1] / v_shape[-1]
                                v = v * scale**0.25
                        if isinstance(layer, FeedForward):
                            count = 1
                            if layer.use_bias:
                                count += 1
                            if self.hidden_act in ['relu', 'leaky_relu']:
                                count -= 2
                            if i < count:
                                v *= np.sqrt(1.0 * w_shape[-1] / v_shape[-1])
                            else:
                                v *= np.sqrt(1.0 * v_shape[0] / w_shape[0])

                    weight_value_pairs.append((w, v))

        K.batch_set_value(weight_value_pairs)

    def save_weights_as_checkpoint(self, filename, mapping=None, dtype=None):
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
                    variable, value = self.create_variable(name, value, dtype)
                    all_variables.append(variable)
                    all_values.append(value)
            with tf.Session() as sess:
                K.batch_set_value(zip(all_variables, all_values))
                saver = tf.train.Saver()
                saver.save(sess, filename)


class LM_Mask(object):
    """定义下三角Attention Mask（语言模型用）
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        if self.attention_bias is None:

            def lm_mask(s):
                seq_len = K.shape(s)[1]
                idxs = K.arange(0, seq_len)
                mask = idxs[None, :] <= idxs[:, None]
                mask = K.cast(mask, K.floatx())
                return -(1 - mask[None, None]) * 1e12

            self.attention_bias = self.apply(
                inputs=self.inputs[0],
                layer=Lambda,
                function=lm_mask,
                name='Attention-LM-Mask'
            )

        return self.attention_bias


class UniLM_Mask(object):
    """定义UniLM的Attention Mask（Seq2Seq模型用）
    其中source和target的分区，由segment_ids来表示。
    UniLM: https://arxiv.org/abs/1905.03197
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        if self.attention_bias is None:

            def unilm_mask(s):
                idxs = K.cumsum(s, axis=1)
                mask = idxs[:, None, :] <= idxs[:, :, None]
                mask = K.cast(mask, K.floatx())
                return -(1 - mask[:, None]) * 1e12

            self.attention_bias = self.apply(
                inputs=self.inputs[1],
                layer=Lambda,
                function=unilm_mask,
                name='Attention-UniLM-Mask'
            )

        return self.attention_bias


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
        hierarchical_position=None,  # 是否层次分解位置编码
        custom_position_ids=False,  # 是否自行传入位置id
        shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
        **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hierarchical_position = hierarchical_position
        self.custom_position_ids = custom_position_ids
        self.shared_segment_embeddings = shared_segment_embeddings
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
            if self.shared_segment_embeddings:
                name = 'Embedding-Token'
            else:
                name = 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name
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
            hierarchical=self.hierarchical_position,
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
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_bias': None}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
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

    def create_variable(self, name, value, dtype=None):
        """在tensorflow中创建一个变量
        """
        if name == 'cls/seq_relationship/output_weights':
            value = value.T
        return super(BERT, self).create_variable(name, value, dtype)

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
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_bias': None}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
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
            if self.shared_segment_embeddings:
                name = 'Embedding-Token'
            else:
                name = 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=2,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name
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
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi, x = x, [x, x, x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'typical_relative'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(3, attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
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

            x = inputs
            self.position_bias = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbedding,
                input_dim=2 * 64 + 1,
                output_dim=self.attention_key_size,
                embeddings_initializer='Sinusoidal',
                name='Embedding-Relative-Position',
                trainable=False
            )

        return self.position_bias


class RoFormer(NEZHA):
    """旋转式位置编码的BERT模型
    链接：https://kexue.fm/archives/8265
    """
    def apply_main_layers(self, inputs, index):
        """RoFormer的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi, x = x, [x, x, x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(3, attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
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
        """Sinusoidal位置编码（直接返回）
        """
        if self.position_bias is None:

            x = inputs
            self.position_bias = self.apply(
                inputs=x,
                layer=SinusoidalPositionEmbedding,
                output_dim=self.attention_key_size,
                merge_mode='zero',
                name='Embedding-Rotary-Position'
            )

        return self.position_bias


class ELECTRA(BERT):
    """Google推出的ELECTRA模型
    链接：https://arxiv.org/abs/2003.10555
    """
    @insert_arguments(with_discriminator=False)
    @delete_arguments('with_pool', 'with_mlm')
    def __init__(
        self,
        max_position,  # 序列最大长度
        **kwargs  # 其余参数
    ):
        super(ELECTRA, self).__init__(max_position, **kwargs)

    def apply_final_layers(self, inputs):
        x = inputs

        if self.with_discriminator:
            if self.with_discriminator is True:
                final_activation = 'sigmoid'
            else:
                final_activation = self.with_discriminator
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name='Discriminator-Dense'
            )
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=1,
                activation=final_activation,
                kernel_initializer=self.initializer,
                name='Discriminator-Prediction'
            )

        return x

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(ELECTRA, self).load_variable(checkpoint, name)
        if name == 'electra/embeddings/word_embeddings':
            return self.load_embeddings(variable)
        else:
            return variable

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
        mapping['Discriminator-Dense'] = [
            'discriminator_predictions/dense/kernel',
            'discriminator_predictions/dense/bias',
        ]
        mapping['Discriminator-Prediction'] = [
            'discriminator_predictions/dense_1/kernel',
            'discriminator_predictions/dense_1/bias',
        ]
        return mapping


class GPT(LM_Mask, BERT):
    """构建GPT模型
    链接：https://github.com/openai/finetune-transformer-lm
    """
    @insert_arguments(final_activation='softmax')
    @delete_arguments('with_pool', 'with_mlm')
    def __init__(self, **kwargs):
        super(GPT, self).__init__(**kwargs)

    def apply_embeddings(self, inputs):
        """GPT的embedding是token、position、segment三者embedding之和
        跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
        if self.custom_position_ids:
            p = inputs.pop(0)
        else:
            p = None

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
            if self.shared_segment_embeddings:
                name = 'Embedding-Token'
            else:
                name = 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name
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
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position'
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

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs

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
        variable = super(GPT, self).load_variable(checkpoint, name)
        if name == 'gpt/embeddings/word_embeddings':
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        """映射到TF版GPT权重格式
        """
        mapping = super(GPT, self).variable_mapping()
        mapping = {
            k: [
                i.replace('bert/', 'gpt/').replace('encoder', 'transformer')
                for i in v
            ]
            for k, v in mapping.items()
        }
        return mapping


class GPT2(GPT):
    """构建GPT2模型
    链接: https://github.com/openai/gpt-2
    """
    def get_inputs(self):
        """GPT2的输入是token_ids
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
        )
        return x_in

    def apply_embeddings(self, inputs):
        """GPT2的embedding是token、position两者embedding之和
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
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            name='Embedding-Position'
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
        """GPT2的主体是基于Self-Attention的模块
        顺序：LN --> Att  --> Add --> LN --> FFN --> Add
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )
        x = self.apply(
            inputs=[x, x, x, attention_mask],
            layer=MultiHeadAttention,
            arguments={'a_bias': True},
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
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
            name='%s-Norm' % feed_forward_name
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
        return x

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            epsilon=1e-5,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Output-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Output-Dropout'
        )
        x = super(GPT2, self).apply_final_layers(x)

        return x

    def variable_mapping(self):
        """映射到TF版GPT2权重格式
        """
        mapping = super(GPT2, self).variable_mapping()
        mapping = {
            k: [i.replace('output/LayerNorm', 'input/LayerNorm') for i in v]
            for k, v in mapping.items()
        }
        mapping['Output-Norm'] = [
            'gpt/output/LayerNorm/beta',
            'gpt/output/LayerNorm/gamma',
        ]

        return mapping


class GPT2_ML(GPT):
    """构建GPT2_ML模型
    链接: https://github.com/imcaspar/gpt2-ml
    注意：GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的
         原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。
    """
    def get_inputs(self):
        """GPT2_ML的输入是token_ids
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
            hierarchical=self.hierarchical_position,
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
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x, attention_mask], {'a_bias': True}

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
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

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(GPT2_ML, self).load_variable(checkpoint, name)
        if name == 'newslm/embeddings/word_embed':
            return self.load_embeddings(variable)
        else:
            return variable

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
    注意T5有两个版本，一开始放出来的版本称为t5.1.0，而后来放出了一个升级
    版本称为t5.1.1，两者结构略有不同，包括后来放出来的多国语言版T5也采用
    了t5.1.1的结构。
    t5.1.0: https://github.com/google-research/text-to-text-transfer-transformer
    t5.1.1: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md#t511
    multilingual-t5: https://github.com/google-research/multilingual-t5
    """
    @insert_arguments(version='t5.1.0')
    def __init__(self, **kwargs):
        super(T5_Base, self).__init__(**kwargs)

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(T5_Base, self).load_variable(checkpoint, name)
        if name == 'shared/embedding':
            return self.load_embeddings(variable)
        elif name == 'decoder/logits/kernel':
            return self.load_embeddings(variable.T).T
        elif 'relative_attention_bias' in name:
            return variable.T
        else:
            return variable

    def create_variable(self, name, value, dtype=None):
        """在tensorflow中创建一个变量
        """
        if 'relative_attention_bias' in name:
            value = value.T
        return super(T5_Base, self).create_variable(name, value, dtype)

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

        if self.version == 't5.1.1':
            mapping['Encoder-Output-Norm'] = ['encoder/rms_norm/scale']
            mapping['Decoder-Output-Norm'] = ['decoder/rms_norm/scale']
            mapping['Decoder-Output-LM'] = ['decoder/logits/kernel']
            mapping = {
                k: [i.replace('layer_norm', 'rms_norm') for i in v]
                for k, v in mapping.items()
            }
            for i in range(self.num_hidden_layers):
                for layer in [
                    'Encoder-Transformer-%d-FeedForward' % i,
                    'Decoder-Transformer-%d-FeedForward' % i
                ]:
                    mapping[layer] = [
                        mapping[layer][0][:-7] + '_0' + mapping[layer][0][-7:],
                        mapping[layer][0][:-7] + '_1' + mapping[layer][0][-7:],
                        mapping[layer][1]
                    ]

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
        attention_mask = self.compute_attention_bias(index)
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
            out_dim=self.hidden_size,
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
        x = self.apply(
            inputs=x,
            layer=Lambda,
            function=sequence_masking,
            name='Encoder-Output-Masked'
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


class T5_Decoder(LM_Mask, T5_Base):
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

        c = self.apply(
            inputs=c, layer=Masking, mask_value=0.0, name='Masked-Context'
        )
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
        attention_mask = self.compute_attention_bias(index)
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
                'a_bias': True,
                'p_bias': 't5_relative'
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
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
                'a_bias': None,
                'p_bias': 't5_relative'
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
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
            function=lambda x: x / self.hidden_size**0.5,
            mask=lambda i, m: m,
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
            lm_activation = 'softmax' if self.with_lm is True else self.with_lm
            if self.version == 't5.1.0':
                x = self.apply(
                    inputs=x,
                    layer=Embedding,
                    arguments={'mode': 'dense'},
                    name='Embedding-Token'
                )
                x = self.apply(
                    inputs=x,
                    layer=Activation,
                    activation=lm_activation,
                    name='Dencoder-Output-LM-Activation'
                )
            else:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.vocab_size,
                    activation=lm_activation,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-LM'
                )

        return x

    def compute_attention_bias(self, inputs=None):
        """修改LM Mask的序列长度（从 self.inputs[0] 改为 self.inputs[1] ）
        """
        old_inputs = self.inputs[:]
        self.inputs = [old_inputs[1]]
        mask = super(T5_Decoder, self).compute_attention_bias(inputs)
        self.inputs = old_inputs
        return mask

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
    class LanguageModel(LM_Mask, BaseModel):
        """带下三角Attention Mask的派生模型
        """
        def __init__(self, *args, **kwargs):
            super(LanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True

    return LanguageModel


def extend_with_unified_language_model(BaseModel):
    """添加UniLM的Attention Mask（Seq2Seq模型用）
    """
    class UnifiedLanguageModel(UniLM_Mask, BaseModel):
        """带UniLM的Attention Mask的派生模型
        UniLM: https://arxiv.org/abs/1905.03197
        """
        def __init__(self, *args, **kwargs):
            super(UnifiedLanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True

    return UnifiedLanguageModel


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
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)

    models = {
        'bert': BERT,
        'albert': ALBERT,
        'albert_unshared': ALBERT_Unshared,
        'roberta': BERT,
        'nezha': NEZHA,
        'roformer': RoFormer,
        'electra': ELECTRA,
        'gpt': GPT,
        'gpt2': GPT2,
        'gpt2_ml': GPT2_ML,
        't5': T5,
        't5_encoder': T5_Encoder,
        't5_decoder': T5_Decoder,
        't5.1.0': T5,
        't5.1.0_encoder': T5_Encoder,
        't5.1.0_decoder': T5_Decoder,
        't5.1.1': T5,
        't5.1.1_encoder': T5_Encoder,
        't5.1.1_decoder': T5_Decoder,
    }

    if is_string(model):
        model = model.lower()
        MODEL = models[model]
        if model.startswith('t5.1.1'):
            configs['version'] = 't5.1.1'
    else:
        MODEL = model

    application = application.lower()
    if application in ['lm', 'unilm'] and model in ['electra', 't5']:
        raise ValueError(
            '"%s" model can not be used as "%s" application.\n' %
            (model, application)
        )

    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unified_language_model(MODEL)

    transformer = MODEL(**configs)
    transformer.build(**configs)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer.model
    else:
        return transformer
