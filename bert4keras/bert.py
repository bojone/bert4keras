#! -*- coding: utf-8 -*-
# 主要模型

from .layers import *
from functools import partial
import json

gelu_version = 'erf'


def get_gelu():
    if gelu_version == 'erf':
        return gelu_erf
    else:
        return gelu_tanh


def set_gelu(version):
    """提供gelu版本切换功能，默认为Erf版本
    """
    global gelu_version
    gelu_version = version


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
            embedding_size=None,  # 是否指定embedding_size
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
        if embedding_size:
            self.embedding_size = embedding_size
        else:
            self.embedding_size = hidden_size
        self.with_mlm = with_mlm
        if hidden_act == 'gelu':
            self.hidden_act = get_gelu()
        else:
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
        sequence_mask = K.cast(K.greater(x, 0), 'float32')

        # Embedding部分
        if self.embedding_size == self.hidden_size:
            token_embedding = Embedding(input_dim=self.vocab_size,
                                        output_dim=self.embedding_size,
                                        name='Embedding-Token')
        else:
            token_embedding = FactorizedEmbedding(
                input_dim=self.vocab_size,
                hidden_dim=self.embedding_size,
                output_dim=self.hidden_size,
                name='Embedding-Token')
        x = token_embedding(x)
        s = Embedding(input_dim=2,
                      output_dim=self.hidden_size,
                      name='Embedding-Segment')(s)
        x = Add(name='Embedding-Token-Segment')([x, s])
        x = PositionEmbedding(input_dim=self.max_position_embeddings,
                              output_dim=self.hidden_size,
                              name='Embedding-Position')(x)
        if self.dropout_rate > 0:
            x = Dropout(rate=self.dropout_rate, name='Embedding-Dropout')(x)
        x = LayerNormalization(name='Embedding-Norm')(x)
        layers = None

        # 主要Transformer部分
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

        if self.with_mlm:
            # Masked Language Model 部分
            x = Dense(self.hidden_size,
                      activation=self.hidden_act,
                      name='MLM-Dense')(x)
            x = LayerNormalization(name='MLM-Norm')(x)
            x = EmbeddingDense(token_embedding, name='MLM-Proba')(x)

        if self.additional_outputs:
            self.model = Model([x_in, s_in], [x] + self.additional_outputs)
        else:
            self.model = Model([x_in, s_in], x)

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
                                   name=attention_name),
                Dropout(rate=self.dropout_rate,
                        name='%s-Dropout' % attention_name),
                Add(name='%s-Add' % attention_name),
                LayerNormalization(name='%s-Norm' % attention_name),
                FeedForward(units=self.intermediate_size,
                            activation=self.hidden_act,
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
        x = layers[0]([x, x, x], v_mask=sequence_mask, a_mask=attention_mask)
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

    def load_weights_from_checkpoint(self, checkpoint_file):
        """从预训练好的Bert的checkpoint中加载权重
        """
        model = self.model
        loader = partial(tf.train.load_variable, checkpoint_file)

        if self.keep_words is None:
            keep_words = slice(0, None)
        else:
            keep_words = self.keep_words

        if self.embedding_size == self.hidden_size:
            model.get_layer(name='Embedding-Token').set_weights([
                loader('bert/embeddings/word_embeddings')[keep_words],
            ])
        else:
            model.get_layer(name='Embedding-Token').set_weights([
                loader('bert/embeddings/word_embeddings')[keep_words],
                loader('bert/embeddings/word_embeddings_2'),
            ])
        model.get_layer(name='Embedding-Position').set_weights([
            loader('bert/embeddings/position_embeddings'),
        ])
        model.get_layer(name='Embedding-Segment').set_weights([
            loader('bert/embeddings/token_type_embeddings'),
        ])
        model.get_layer(name='Embedding-Norm').set_weights([
            loader('bert/embeddings/LayerNorm/gamma'),
            loader('bert/embeddings/LayerNorm/beta'),
        ])

        for i in range(self.num_hidden_layers):
            try:
                model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1))
            except ValueError as e:
                continue
            try:
                layer_name = 'layer_%d' % i
                loader('bert/encoder/%s/attention/self/query/kernel' % layer_name)
            except:
                layer_name = 'layer_shared'
                loader('bert/encoder/%s/attention/self/query/kernel' % layer_name)
            model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
                loader('bert/encoder/%s/attention/self/query/kernel' % layer_name),
                loader('bert/encoder/%s/attention/self/query/bias' % layer_name),
                loader('bert/encoder/%s/attention/self/key/kernel' % layer_name),
                loader('bert/encoder/%s/attention/self/key/bias' % layer_name),
                loader('bert/encoder/%s/attention/self/value/kernel' % layer_name),
                loader('bert/encoder/%s/attention/self/value/bias' % layer_name),
                loader('bert/encoder/%s/attention/output/dense/kernel' % layer_name),
                loader('bert/encoder/%s/attention/output/dense/bias' % layer_name),
            ])
            model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
                loader('bert/encoder/%s/attention/output/LayerNorm/gamma' % layer_name),
                loader('bert/encoder/%s/attention/output/LayerNorm/beta' % layer_name),
            ])
            model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
                loader('bert/encoder/%s/attention/output/LayerNorm/gamma' % layer_name),
                loader('bert/encoder/%s/attention/output/LayerNorm/beta' % layer_name),
            ])
            model.get_layer(
                name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
                    loader('bert/encoder/%s/intermediate/dense/kernel' % layer_name),
                    loader('bert/encoder/%s/intermediate/dense/bias' % layer_name),
                    loader('bert/encoder/%s/output/dense/kernel' % layer_name),
                    loader('bert/encoder/%s/output/dense/bias' % layer_name),
                ])
            model.get_layer(
                name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
                    loader('bert/encoder/%s/output/LayerNorm/gamma' % layer_name),
                    loader('bert/encoder/%s/output/LayerNorm/beta' % layer_name),
                ])

        if self.with_mlm:
            model.get_layer(name='MLM-Dense').set_weights([
                loader('cls/predictions/transform/dense/kernel'),
                loader('cls/predictions/transform/dense/bias'),
            ])
            model.get_layer(name='MLM-Norm').set_weights([
                loader('cls/predictions/transform/LayerNorm/gamma'),
                loader('cls/predictions/transform/LayerNorm/beta'),
            ])
            model.get_layer(name='MLM-Proba').set_weights([
                loader('cls/predictions/output_bias')[keep_words],
            ])


class Bert4Seq2seq(BertModel):
    """用来做seq2seq任务的Bert
    """
    def __init__(self, *args, **kwargs):
        super(Bert4Seq2seq, self).__init__(*args, **kwargs)
        self.with_mlm = True
        self.attention_mask = None

    def compute_attention_mask(self, layer_id, segment_ids):
        """为seq2seq采用特定的attention mask
        """
        if self.attention_mask is None:
            s = segment_ids
            seq_len = K.shape(s)[1]
            ones = K.ones((1, self.num_attention_heads, seq_len, seq_len))
            a_mask = tf.matrix_band_part(ones, -1, 0)
            s_ex12 = K.expand_dims(K.expand_dims(s, 1), 2)
            s_ex13 = K.expand_dims(K.expand_dims(s, 1), 3)
            a_mask = (1 - s_ex13) * (1 - s_ex12) + s_ex13 * a_mask
            a_mask = K.reshape(a_mask, (-1, seq_len, seq_len))
            self.attention_mask = a_mask

        return self.attention_mask


def load_pretrained_model(config_path,
                          checkpoint_file,
                          with_mlm=False,
                          seq2seq=False,
                          keep_words=None,
                          albert=False):
    """根据配置文件和checkpoint文件来加载模型
    """
    config = json.load(open(config_path))

    if seq2seq:
        Bert = Bert4Seq2seq
    else:
        Bert = BertModel

    bert = Bert(vocab_size=config['vocab_size'],
                max_position_embeddings=config['max_position_embeddings'],
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_hidden_layers'],
                num_attention_heads=config['num_attention_heads'],
                intermediate_size=config['intermediate_size'],
                hidden_act=config['hidden_act'],
                dropout_rate=config['hidden_dropout_prob'],
                embedding_size=config.get('embedding_size'),
                with_mlm=with_mlm,
                keep_words=keep_words,
                block_sharing=albert)

    bert.build()
    bert.load_weights_from_checkpoint(checkpoint_file)

    return bert.model
