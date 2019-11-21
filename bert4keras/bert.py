#! -*- coding: utf-8 -*-
# 主要模型

import numpy as np
from bert4keras.layers import *
from collections import OrderedDict
import json


Model = keras.models.Model


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
            num_feed_forward_groups=1,  # Feed Forward部分是否使用分组Dense
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            keep_words=None,  # 要保留的词ID列表
            block_sharing=False,  # 是否共享同一个transformer block
    ):
        if keep_words is None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = len(keep_words)
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        if initializer_range:
            self.initializer_range = initializer_range
        else:
            self.initializer_range = 0.02
        if embedding_size:
            self.embedding_size = embedding_size
        else:
            self.embedding_size = hidden_size
        self.num_feed_forward_groups = num_feed_forward_groups
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hidden_act = hidden_act
        self.keep_words = keep_words
        self.block_sharing = block_sharing
        self.additional_outputs = []

    def build(self):
        """Bert模型构建函数
        """
        x_in = Input(shape=(None, ), name='Input-Token')
        s_in = Input(shape=(None, ), name='Input-Segment')
        x, s = x_in, s_in

        # 自行构建Mask
        sequence_mask = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'),
                               name='Sequence-Mask')(x)

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
        x = PositionEmbedding(input_dim=self.max_position_embeddings,
                              output_dim=self.embedding_size,
                              merge_mode='add',
                              embeddings_initializer=self.initializer,
                              name='Embedding-Position')(x)
        x = LayerNormalization(name='Embedding-Norm')(x)
        if self.dropout_rate > 0:
            x = Dropout(rate=self.dropout_rate, name='Embedding-Dropout')(x)
        if self.embedding_size != self.hidden_size:
            x = Dense(units=self.hidden_size,
                      kernel_initializer=self.initializer,
                      name='Embedding-Mapping')(x)

        # 主要Transformer部分
        layers = None
        for i in range(self.num_hidden_layers):
            attention_name = 'Encoder-%d-MultiHeadSelfAttention' % (i + 1)
            feed_forward_name = 'Encoder-%d-FeedForward' % (i + 1)
            x, layers = self.transformer_block(
                inputs=x,
                sequence_mask=sequence_mask,
                attention_mask=self.compute_attention_mask(i, s_in),
                attention_name=attention_name,
                feed_forward_name=feed_forward_name,
                input_layers=layers)
            x = self.post_processing(i, x)
            if not self.block_sharing:
                layers = None

        outputs = [x]

        if self.with_pool:
            # Pooler部分（提取CLS向量）
            x = outputs[0]
            x = Lambda(lambda x: x[:, 0], name='Pooler')(x)
            x = Dense(units=self.hidden_size,
                      activation='tanh',
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
            x = LayerNormalization(name='MLM-Norm')(x)
            x = EmbeddingDense(embedding_name='Embedding-Token',
                               activation=self.with_mlm,
                               name='MLM-Proba')(x)
            outputs.append(x)

        outputs += self.additional_outputs
        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        self.model = Model([x_in, s_in], outputs)

    def transformer_block(self,
                          inputs,
                          sequence_mask,
                          attention_mask=None,
                          attention_name='attention',
                          feed_forward_name='feed-forward',
                          input_layers=None):
        """构建单个Transformer Block
        如果没传入input_layers则新建层；如果传入则重用旧层。
        """
        x = inputs
        if input_layers is None:
            layers = [
                MultiHeadAttention(heads=self.num_attention_heads,
                                   head_size=self.attention_head_size,
                                   kernel_initializer=self.initializer,
                                   name=attention_name),
                Dropout(rate=self.dropout_rate,
                        name='%s-Dropout' % attention_name),
                Add(name='%s-Add' % attention_name),
                LayerNormalization(name='%s-Norm' % attention_name),
                FeedForward(units=self.intermediate_size,
                            groups=self.num_feed_forward_groups,
                            activation=self.hidden_act,
                            kernel_initializer=self.initializer,
                            name=feed_forward_name),
                Dropout(rate=self.dropout_rate,
                        name='%s-Dropout' % feed_forward_name),
                Add(name='%s-Add' % feed_forward_name),
                LayerNormalization(name='%s-Norm' % feed_forward_name),
            ]
        else:
            layers = input_layers
        # Self Attention
        xi = x
        if attention_mask is None:
            x = layers[0]([x, x, x, sequence_mask], v_mask=True)
        elif attention_mask is 'history_only':
            x = layers[0]([x, x, x, sequence_mask], v_mask=True)
        else:
            x = layers[0]([x, x, x, sequence_mask, attention_mask],
                          v_mask=True,
                          a_mask=True)
        if self.dropout_rate > 0:
            x = layers[1](x)
        x = layers[2]([xi, x])
        x = layers[3](x)
        # Feed Forward
        xi = x
        x = layers[4](x)
        if self.dropout_rate > 0:
            x = layers[5](x)
        x = layers[6]([xi, x])
        x = layers[7](x)
        return x, layers

    def compute_attention_mask(self, layer_id, segment_ids):
        """定义每一层的Attention Mask，来实现不同的功能
        """
        return None

    def post_processing(self, layer_id, inputs):
        """自定义每一个block的后处理操作
        """
        return inputs

    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(
            stddev=self.initializer_range)

    def variable_mapping(self, variable_names):
        """构建Keras层与checkpoint的变量名之间的映射表
        """
        mapping = OrderedDict()

        mapping['Embedding-Token'] = ['bert/embeddings/word_embeddings']
        mapping['Embedding-Position'] = ['bert/embeddings/position_embeddings']
        mapping['Embedding-Segment'] = ['bert/embeddings/token_type_embeddings']
        mapping['Embedding-Norm'] = [
            'bert/embeddings/LayerNorm/gamma',
            'bert/embeddings/LayerNorm/beta',
        ]

        if self.embedding_size != self.hidden_size:
            mapping['Embedding-Mapping'] = [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ]

        for i in range(self.num_hidden_layers):
            try:
                self.model.get_layer('Encoder-%d-MultiHeadSelfAttention' % (i + 1))
            except ValueError:
                continue
            if ('bert/encoder/layer_%d/attention/self/query/kernel' % i) in variable_names:
                block_name = 'layer_%d' % i
            else:
                block_name = 'transformer/group_0/inner_group_0'

            mapping['Encoder-%d-MultiHeadSelfAttention' % (i + 1)] = [
                'bert/encoder/%s/attention/self/query/kernel' % block_name,
                'bert/encoder/%s/attention/self/query/bias' % block_name,
                'bert/encoder/%s/attention/self/key/kernel' % block_name,
                'bert/encoder/%s/attention/self/key/bias' % block_name,
                'bert/encoder/%s/attention/self/value/kernel' % block_name,
                'bert/encoder/%s/attention/self/value/bias' % block_name,
                'bert/encoder/%s/attention/output/dense/kernel' % block_name,
                'bert/encoder/%s/attention/output/dense/bias' % block_name,
            ]
            mapping['Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)] = [
                'bert/encoder/%s/attention/output/LayerNorm/gamma' % block_name,
                'bert/encoder/%s/attention/output/LayerNorm/beta' % block_name,
            ]
            mapping['Encoder-%d-FeedForward' % (i + 1)] = [
                'bert/encoder/%s/intermediate/dense/kernel' % block_name,
                'bert/encoder/%s/intermediate/dense/bias' % block_name,
                'bert/encoder/%s/output/dense/kernel' % block_name,
                'bert/encoder/%s/output/dense/bias' % block_name,
            ]
            mapping['Encoder-%d-FeedForward-Norm' % (i + 1)] = [
                'bert/encoder/%s/output/LayerNorm/gamma' % block_name,
                'bert/encoder/%s/output/LayerNorm/beta' % block_name,
            ]

        if self.with_pool:
            mapping['Pooler-Dense'] = [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ]
            if self.with_nsp:
                mapping['NSP-Proba'] = [
                    'cls/seq_relationship/output_weights',
                    'cls/seq_relationship/output_bias',
                ]

        if self.with_mlm:
            mapping['MLM-Dense'] = [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ]
            mapping['MLM-Norm'] = [
                'cls/predictions/transform/LayerNorm/gamma',
                'cls/predictions/transform/LayerNorm/beta',
            ]
            mapping['MLM-Proba'] = ['cls/predictions/output_bias']

        return mapping

    def load_weights_from_checkpoint(self, checkpoint_file, mapping=None):
        """从预训练好的Bert的checkpoint中加载权重
        为了简化写法，对变量名的匹配引入了一定的模糊匹配能力。
        """
        variable_names = [
            n[0] for n in tf.train.list_variables(checkpoint_file)
            if 'adam' not in n[0]
        ]
        if mapping is None:
            mapping = self.variable_mapping(variable_names)

        def similarity(a, b, n=4):
            # 基于n-grams的jaccard相似度
            a = set([a[i:i + n] for i in range(len(a) - n)])
            b = set([b[i:i + n] for i in range(len(b) - n)])
            a_and_b = a & b
            if not a_and_b:
                return 0.
            a_or_b = a | b
            return 1. * len(a_and_b) / len(a_or_b)

        def load_variable(name):
            # 加载单个变量的函数
            sims = [similarity(name, n) for n in variable_names]
            found_name = variable_names.pop(np.argmax(sims))
            print('==> searching: %s, found name: %s' % (name, found_name))
            variable = tf.train.load_variable(checkpoint_file, found_name)
            if name in [
                'bert/embeddings/word_embeddings',
                'cls/predictions/output_bias',
            ]:
                if self.keep_words is None:
                    return variable
                else:
                    return variable[self.keep_words]
            elif name == 'cls/seq_relationship/output_weights':
                return variable.T
            else:
                return variable

        def load_variables(names):
            # 批量加载的函数
            if not isinstance(names, list):
                names = [names]
            return [load_variable(name) for name in names]

        for layer_name, layer_variable_names in mapping.items():
            weights = load_variables(layer_variable_names)
            self.model.get_layer(layer_name).set_weights(weights)

    def save_weights_as_checkpoint(self, filename, reference, mapping=None):
        """保存模型的权重，跟Bert的checkpoint格式一致
        filename: 要保存的名字；
        reference: 参照的已有的checkpoint。
        """
        variable_names = [
            n[0] for n in tf.train.list_variables(reference)
            if 'adam' not in n[0]
        ]
        if mapping is None:
            mapping = self.variable_mapping(variable_names)

        weights = {}
        for layer_name, layer_variable_names in mapping.items():
            layer_weights = self.model.get_layer(layer_name).get_weights()
            for n, w in zip(layer_variable_names, layer_weights):
                weights[n] = w

        def create_variable(name, value):
            if name == 'cls/seq_relationship/output_weights':
                value = value.T
            return tf.Variable(value, name=name)

        with tf.Graph().as_default():
            for n, w in weights.items():
                create_variable(n, w)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.save(sess, filename)


class Bert4Seq2seq(BertModel):
    """用来做seq2seq任务的Bert
    """
    def __init__(self, *args, **kwargs):
        super(Bert4Seq2seq, self).__init__(*args, **kwargs)
        self.with_pool = False
        self.with_nsp = False
        self.with_mlm = True
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
        self.with_pool = False
        self.with_nsp = False
        self.with_mlm = True
        self.attention_mask = 'history_only'

    def compute_attention_mask(self, layer_id, segment_ids):
        return self.attention_mask


def build_bert_model(config_path,
                     checkpoint_path=None,
                     with_pool=False,
                     with_nsp=False,
                     with_mlm=False,
                     application='encoder',
                     keep_words=None,
                     albert=False,
                     return_keras_model=True):
    """根据配置文件构建bert模型，可选加载checkpoint权重
    """
    config = json.load(open(config_path))
    mapping = {
        'encoder': BertModel,
        'seq2seq': Bert4Seq2seq,
        'lm': Bert4LM,
    }

    assert application in mapping, 'application must be one of %s' % list(mapping.keys())
    Bert = mapping[application]

    bert = Bert(vocab_size=config['vocab_size'],
                max_position_embeddings=config['max_position_embeddings'],
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_hidden_layers'],
                num_attention_heads=config['num_attention_heads'],
                intermediate_size=config['intermediate_size'],
                hidden_act=config['hidden_act'],
                dropout_rate=config['hidden_dropout_prob'],
                initializer_range=config.get('initializer_range'),
                embedding_size=config.get('embedding_size'),
                num_feed_forward_groups=config.get('num_feed_forward_groups'),
                with_pool=with_pool,
                with_nsp=with_nsp,
                with_mlm=with_mlm,
                keep_words=keep_words,
                block_sharing=albert)

    bert.build()

    if checkpoint_path is not None:
        bert.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return bert.model
    else:
        return bert
