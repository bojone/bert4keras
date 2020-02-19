#! -*- coding: utf-8 -*-
# 主要模型

import numpy as np
from bert4keras.layers import *
from collections import OrderedDict
import json


class BertModel(object):
    """构建跟Bert一样结构的Transformer-based模型
    这是一个比较多接口的基础类，然后通过这个基础类衍生出更复杂的模型
    """
    def __init__(
            self,
            vocab_size,  # 词表大小
            max_position_embeddings,  # 序列最大长度
            hidden_size,  # 编码维度
            num_hidden_layers,  # Transformer总层数
            num_attention_heads,  # Attention的头数
            intermediate_size,  # FeedForward的隐层维度
            hidden_act,  # FeedForward隐层的激活函数
            dropout_rate,  # Dropout比例
            initializer_range=None,  # 权重初始化方差
            embedding_size=None,  # 是否指定embedding_size
            max_relative_position=None,  # 非None则使用相对位置编码
            num_feed_forward_groups=1,  # Feed Forward部分是否使用分组Dense
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            keep_tokens=None,  # 要保留的词ID列表
            block_sharing=False,  # 是否共享同一个transformer block
            att_pool_size=None,  # 进行attention之前是否先pooling
            ffn_pool_size=None,  # 输入FFN之前是否先pooling
    ):
        if keep_tokens is None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = len(keep_tokens)
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range or 0.02
        self.embedding_size = embedding_size or hidden_size
        self.max_relative_position = max_relative_position
        self.num_feed_forward_groups = num_feed_forward_groups
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hidden_act = hidden_act
        self.keep_tokens = keep_tokens
        self.block_sharing = block_sharing
        if isinstance(att_pool_size, list):
            self.att_pool_size = att_pool_size
        else:
            self.att_pool_size = [att_pool_size] * num_hidden_layers
        if isinstance(ffn_pool_size, list):
            self.ffn_pool_size = ffn_pool_size
        else:
            self.ffn_pool_size = [ffn_pool_size] * num_hidden_layers
        self.additional_outputs = []

    def build(self,
              position_ids=None,
              layer_norm_cond=None,
              layer_norm_cond_size=None,
              layer_norm_cond_hidden_size=None,
              layer_norm_cond_hidden_act=None,
              additional_input_layers=None):
        """Bert模型构建函数
        layer_norm_*系列参数为实现Conditional Layer Normalization时使用，
        用来实现以“固定长度向量”为条件的条件Bert。
        """
        # 构建输入层
        x_in = Input(shape=(None, ), name='Input-Token')
        s_in = Input(shape=(None, ), name='Input-Segment')
        x, s = input_layers = [x_in, s_in]

        # 条件输入
        if layer_norm_cond is not None:
            z = layer_norm_cond
        elif layer_norm_cond_size is not None:
            z = Input(shape=(layer_norm_cond_size, ), name='LayerNorm-Condition')
            input_layers.append(z)
        else:
            z = None
        layer_norm_cond_hidden_act = layer_norm_cond_hidden_act or 'linear'

        # 补充输入层
        if additional_input_layers is not None:
            if isinstance(additional_input_layers, list):
                input_layers.extend(additional_input_layers)
            else:
                input_layers.append(additional_input_layers)

        # 补充mask
        x = ZeroMasking(name='Sequence-Mask')(x)

        # Embedding部分
        x = Embedding(input_dim=self.vocab_size,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      name='Embedding-Token')(x)
        s = Embedding(input_dim=2,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      name='Embedding-Segment')(s)
        x = Add(name='Embedding-Token-Segment')([x, s])
        if self.max_relative_position is None:
            x = self.filter([x, position_ids])
            x = PositionEmbedding(input_dim=self.max_position_embeddings,
                                  output_dim=self.embedding_size,
                                  merge_mode='add',
                                  embeddings_initializer=self.initializer,
                                  name='Embedding-Position')(x)
        x = LayerNormalization(conditional=(z is not None),
                               hidden_units=layer_norm_cond_hidden_size,
                               hidden_activation=layer_norm_cond_hidden_act,
                               hidden_initializer=self.initializer,
                               name='Embedding-Norm')(self.filter([x, z]))
        if self.dropout_rate > 0:
            x = Dropout(rate=self.dropout_rate, name='Embedding-Dropout')(x)
        if self.embedding_size != self.hidden_size:
            x = Dense(units=self.hidden_size,
                      kernel_initializer=self.initializer,
                      name='Embedding-Mapping')(x)

        # 主要Transformer部分
        layers = None
        for i in range(self.num_hidden_layers):
            attention_name = 'Transformer-%d-MultiHeadSelfAttention' % (i + 1)
            feed_forward_name = 'Transformer-%d-FeedForward' % (i + 1)
            x, layers = self.transformer_block(
                inputs=[x, z],
                attention_mask=self.compute_attention_mask(i, s_in),
                attention_name=attention_name,
                feed_forward_name=feed_forward_name,
                layer_norm_cond_hidden_size=layer_norm_cond_hidden_size,
                layer_norm_cond_hidden_act=layer_norm_cond_hidden_act,
                attention_pool_size=self.att_pool_size[i],
                feed_forward_pool_size=self.ffn_pool_size[i],
                layers=layers)
            if not self.block_sharing:
                layers = None

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
                                   hidden_units=layer_norm_cond_hidden_size,
                                   hidden_activation=layer_norm_cond_hidden_act,
                                   hidden_initializer=self.initializer,
                                   name='MLM-Norm')(self.filter([x, z]))
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = EmbeddingDense(embedding_name='Embedding-Token',
                               activation=mlm_activation,
                               name='MLM-Proba')(x)
            outputs.append(x)

        outputs += self.additional_outputs
        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        self.model = keras.models.Model(input_layers, outputs)

    def transformer_block(self,
                          inputs,
                          attention_mask=None,
                          attention_name='attention',
                          feed_forward_name='feed-forward',
                          layer_norm_cond_hidden_size=None,
                          layer_norm_cond_hidden_act='linear',
                          attention_pool_size=None,
                          feed_forward_pool_size=None,
                          layers=None):
        """构建单个Transformer Block
        如果没传入layers则新建层；如果传入则重用旧层。
        """
        x, z = inputs
        layers = layers or [
            MultiHeadAttention(heads=self.num_attention_heads,
                               head_size=self.attention_head_size,
                               kernel_initializer=self.initializer,
                               max_relative_position=self.max_relative_position,
                               pool_size=attention_pool_size,
                               name=attention_name),
            Dropout(rate=self.dropout_rate,
                    name='%s-Dropout' % attention_name),
            Add(name='%s-Add' % attention_name),
            LayerNormalization(conditional=(z is not None),
                               hidden_units=layer_norm_cond_hidden_size,
                               hidden_activation=layer_norm_cond_hidden_act,
                               hidden_initializer=self.initializer,
                               name='%s-Norm' % attention_name),
            FeedForward(units=self.intermediate_size,
                        groups=self.num_feed_forward_groups,
                        activation=self.hidden_act,
                        kernel_initializer=self.initializer,
                        pool_size=feed_forward_pool_size,
                        name=feed_forward_name),
            Dropout(rate=self.dropout_rate,
                    name='%s-Dropout' % feed_forward_name),
            Add(name='%s-Add' % feed_forward_name),
            LayerNormalization(conditional=(z is not None),
                               hidden_units=layer_norm_cond_hidden_size,
                               hidden_activation=layer_norm_cond_hidden_act,
                               hidden_initializer=self.initializer,
                               name='%s-Norm' % feed_forward_name),
        ]
        # Self Attention
        xi, x = x, [x, x, x]
        mask = 'Sequence-Mask'
        if attention_mask is None:
            x = layers[0](x, q_mask=mask, v_mask=mask)
        elif attention_mask is 'history_only':
            x = layers[0](x, q_mask=mask, v_mask=mask, a_mask=True)
        else:
            x.append(attention_mask)
            x = layers[0](x, q_mask=mask, v_mask=mask, a_mask=True)
        if self.dropout_rate > 0:
            x = layers[1](x)
        x = layers[2]([xi, x])
        x = layers[3](self.filter([x, z]))
        # Feed Forward
        xi = x
        x = layers[4](x, mask=mask)
        if self.dropout_rate > 0:
            x = layers[5](x)
        x = layers[6]([xi, x])
        x = layers[7](self.filter([x, z]))
        return x, layers

    def compute_attention_mask(self, layer_id, segment_ids):
        """定义每一层的Attention Mask，来实现不同的功能
        """
        return None

    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(
            stddev=self.initializer_range)

    def filter(self, inputs):
        """将list中的None过滤掉
        """
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            inputs = inputs[0]

        return inputs

    def variable_mapping(self, reference='bert'):
        """构建Keras层与checkpoint的变量名之间的映射表
        """
        mapping = [
            'bert/embeddings/word_embeddings',
            'bert/embeddings/token_type_embeddings',
        ]
        if self.max_relative_position is None:
            mapping.extend(['bert/embeddings/position_embeddings'])

        mapping.extend([
            'bert/embeddings/LayerNorm/gamma',
            'bert/embeddings/LayerNorm/beta',
        ])

        if self.embedding_size != self.hidden_size:
            mapping.extend([
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ])

        if reference == 'albert':
            block_weight_names = [
                'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel',
                'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias',
                'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel',
                'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias',
                'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel',
                'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias',
                'bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel',
                'bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias',
                'bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma',
                'bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta',
                'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel',
                'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias',
                'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel',
                'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias',
                'bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma',
                'bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta',
            ]

        if not self.block_sharing and reference != 'albert':
            for i in range(self.num_hidden_layers):
                block_name = 'layer_%d' % i
                mapping.extend([
                    'bert/encoder/%s/attention/self/query/kernel' % block_name,
                    'bert/encoder/%s/attention/self/query/bias' % block_name,
                    'bert/encoder/%s/attention/self/key/kernel' % block_name,
                    'bert/encoder/%s/attention/self/key/bias' % block_name,
                    'bert/encoder/%s/attention/self/value/kernel' % block_name,
                    'bert/encoder/%s/attention/self/value/bias' % block_name,
                    'bert/encoder/%s/attention/output/dense/kernel' % block_name,
                    'bert/encoder/%s/attention/output/dense/bias' % block_name,
                    'bert/encoder/%s/attention/output/LayerNorm/gamma' % block_name,
                    'bert/encoder/%s/attention/output/LayerNorm/beta' % block_name,
                    'bert/encoder/%s/intermediate/dense/kernel' % block_name,
                    'bert/encoder/%s/intermediate/dense/bias' % block_name,
                    'bert/encoder/%s/output/dense/kernel' % block_name,
                    'bert/encoder/%s/output/dense/bias' % block_name,
                    'bert/encoder/%s/output/LayerNorm/gamma' % block_name,
                    'bert/encoder/%s/output/LayerNorm/beta' % block_name,
                ])
        elif not self.block_sharing and reference == 'albert':
            mapping.extend(block_weight_names * self.num_hidden_layers)
        else:
            mapping.extend(block_weight_names)

        if self.with_pool or self.with_nsp:
            mapping.extend([
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ])
            if self.with_nsp:
                mapping.extend([
                    'cls/seq_relationship/output_weights',
                    'cls/seq_relationship/output_bias',
                ])

        if self.with_mlm:
            mapping.extend([
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
                'cls/predictions/transform/LayerNorm/gamma',
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/output_bias',
            ])

        return mapping

    def load_weights_from_checkpoint(self,
                                     checkpoint_file,
                                     reference='bert',
                                     mapping=None):
        """从预训练好的Bert的checkpoint中加载权重
        """
        if mapping is None:
            mapping = self.variable_mapping(reference)

        def load_variable(name):
            # 加载单个变量的函数
            variable = tf.train.load_variable(checkpoint_file, name)
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

        values = [load_variable(name) for name in mapping]

        bert_layer_names = [
            'Embedding-Token',
            'Embedding-Segment',
            'Embedding-Position',
            'Embedding-Norm',
            'Embedding-Mapping',
        ]

        for i in range(self.num_hidden_layers):
            bert_layer_names.extend([
                'Transformer-%d-MultiHeadSelfAttention' % (i + 1),
                'Transformer-%d-MultiHeadSelfAttention-Norm' % (i + 1),
                'Transformer-%d-FeedForward' % (i + 1),
                'Transformer-%d-FeedForward-Norm' % (i + 1),
            ])

        bert_layer_names.extend([
            'Pooler-Dense',
            'NSP-Proba',
            'MLM-Dense',
            'MLM-Norm',
            'MLM-Proba',
        ])

        weights = []
        for layer_name in bert_layer_names:
            try:
                layer = self.model.get_layer(layer_name)
                layer_weights = layer.trainable_weights
            except:
                continue
            if 'Norm' in layer.name:
                layer_weights = layer_weights[:2]
            weights.extend(layer_weights)

        if len(weights) != len(values):
            raise ValueError(
                'Expecting %s weights, but provide a list of %s weights.' %
                (len(weights), len(values)))

        K.batch_set_value(zip(weights, values))

    def save_weights_as_checkpoint(self,
                                   filename,
                                   reference='bert',
                                   mapping=None,
                                   write_meta_graph=False):
        """保存模型的权重，跟Bert的checkpoint格式一致
        """
        weights = self.model.get_weights()

        if mapping is None:
            mapping = self.variable_mapping(reference)

        def create_variable(name, value):
            if name == 'cls/seq_relationship/output_weights':
                value = value.T
            return tf.Variable(value, name=name)

        with tf.Graph().as_default():
            for n, w in zip(mapping, weights):
                create_variable(n, w)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.save(sess, filename, write_meta_graph=write_meta_graph)


class Bert4Seq2seq(BertModel):
    """用来做seq2seq任务的Bert
    """
    def __init__(self, *args, **kwargs):
        super(Bert4Seq2seq, self).__init__(*args, **kwargs)
        self.with_mlm = self.with_mlm or True
        self.attention_mask = None

    def compute_attention_mask(self, layer_id, segment_ids):
        """为seq2seq采用特定的attention mask
        """
        if self.attention_mask is None:

            def seq2seq_attention_mask(s):
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
                seq2seq_attention_mask,
                name='Attention-Mask')(segment_ids)

        return self.attention_mask


class Bert4LM(BertModel):
    """用来做语言模型任务的Bert
    """
    def __init__(self, *args, **kwargs):
        super(Bert4LM, self).__init__(*args, **kwargs)
        self.with_mlm = self.with_mlm or True
        self.attention_mask = 'history_only'

    def compute_attention_mask(self, layer_id, segment_ids):
        return self.attention_mask


def build_bert_model(config_path,
                     checkpoint_path=None,
                     with_pool=False,
                     with_nsp=False,
                     with_mlm=False,
                     model='bert',
                     application='encoder',
                     keep_tokens=None,
                     attention_mask=None,
                     position_ids=None,
                     layer_norm_cond=None,
                     layer_norm_cond_size=None,
                     layer_norm_cond_hidden_size=None,
                     layer_norm_cond_hidden_act=None,
                     additional_input_layers=None,
                     att_pool_size=None,
                     ffn_pool_size=None,
                     return_keras_model=True):
    """根据配置文件构建bert模型，可选加载checkpoint权重
    """
    config = json.load(open(config_path))
    model, application = model.lower(), application.lower()

    applications = {
        'encoder': BertModel,
        'seq2seq': Bert4Seq2seq,
        'lm': Bert4LM,
    }
    if application not in applications:
        raise ValueError('application must be one of ' +
                         str(list(applications.keys())))

    Bert = applications[application]
    
    if attention_mask is not None:
        class Bert(Bert):
            def compute_attention_mask(self, layer_id, segment_ids):
                return attention_mask

    bert = Bert(vocab_size=config['vocab_size'],
                max_position_embeddings=config.get('max_position_embeddings'),
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_hidden_layers'],
                num_attention_heads=config['num_attention_heads'],
                intermediate_size=config['intermediate_size'],
                hidden_act=config['hidden_act'],
                dropout_rate=config['hidden_dropout_prob'],
                initializer_range=config.get('initializer_range'),
                embedding_size=config.get('embedding_size'),
                max_relative_position=(64 if model == 'nezha' else None),
                num_feed_forward_groups=config.get('num_feed_forward_groups'),
                with_pool=with_pool,
                with_nsp=with_nsp,
                with_mlm=with_mlm,
                keep_tokens=keep_tokens,
                block_sharing=(model == 'albert'),
                att_pool_size=att_pool_size,
                ffn_pool_size=ffn_pool_size)

    bert.build(position_ids=position_ids,
               layer_norm_cond=layer_norm_cond,
               layer_norm_cond_size=layer_norm_cond_size,
               layer_norm_cond_hidden_size=layer_norm_cond_hidden_size,
               layer_norm_cond_hidden_act=layer_norm_cond_hidden_act,
               additional_input_layers=additional_input_layers)

    if checkpoint_path is not None:
        if model[:6] == 'albert':
            reference = 'albert'
        else:
            reference = 'bert'
        bert.load_weights_from_checkpoint(checkpoint_path, reference)

    if return_keras_model:
        return bert.model
    else:
        return bert
