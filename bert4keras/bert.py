#! -*- coding: utf-8 -*-
# 主要模型

from layers import *
from functools import partial
import json


def get_bert_model(vocab_size, max_position_embeddings, hidden_size,
                   num_hidden_layers, num_attention_heads, intermediate_size,
                   hidden_act, dropout_rate, seq2seq=False):
    """构建跟Bert一样结构的Transformer-based模型
    如果是seq2seq=True，则进行特殊的mask，使得它可以直接用于seq2seq用途
    """
    attention_head_size = hidden_size // num_attention_heads

    if hidden_act == 'gelu':
        hidden_act = gelu

    x_in = Input(shape=(None, ), name='Input-Token')
    s_in = Input(shape=(None, ), name='Input-Segment')
    x, s = x_in, s_in

    # 自行构建Mask
    mask = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'),
                  name='Input-Mask')(x)
    
    # Attention矩阵的mask，对s_in=1的部分mask掉未来信息
    if seq2seq:
        seq_len = K.shape(s)[1]
        ones = K.ones((1, seq_len, seq_len))
        a_mask = tf.matrix_band_part(ones, -1, 0)
        s_ex1, s_ex2 = K.expand_dims(s, 1), K.expand_dims(s, 2)
        a_mask = (1 - s_ex2) * (1 - s_ex1) + s_ex2 * a_mask
    else:
        a_mask = None
    
    # Embedding部分
    x = Embedding(input_dim=vocab_size,
                  output_dim=hidden_size,
                  name='Embedding-Token')(x)
    s = Embedding(input_dim=2,
                  output_dim=hidden_size,
                  name='Embedding-Segment')(s)
    x = Add(name='Embedding-Token-Segment')([x, s])
    x = PositionEmbedding(input_dim=max_position_embeddings,
                          output_dim=hidden_size,
                          name='Embedding-Position')(x)
    if dropout_rate > 0:
        x = Dropout(rate=dropout_rate, name='Embedding-Dropout')(x)
    x = LayerNormalization(name='Embedding-Norm')(x)

    # Transformer部分
    for i in range(num_hidden_layers):
        attention_name = 'Encoder-%d-MultiHeadSelfAttention' % (i + 1)
        feed_forward_name = 'Encoder-%d-FeedForward' % (i + 1)
        # Self Attention
        xi = x
        x = MultiHeadAttention(heads=num_attention_heads,
                               head_size=attention_head_size,
                               name=attention_name)([x, x, x, mask], mask=a_mask)
        if dropout_rate > 0:
            x = Dropout(rate=dropout_rate,
                        name='%s-Dropout' % attention_name)(x)
        x = Add(name='%s-Add' % attention_name)([xi, x])
        x = LayerNormalization(name='%s-Norm' % attention_name)(x)
        # Feed Forward
        xi = x
        x = FeedForward(units=intermediate_size,
                        activation=hidden_act,
                        name=feed_forward_name)(x)
        if dropout_rate > 0:
            x = Dropout(rate=dropout_rate,
                        name='%s-Dropout' % feed_forward_name)(x)
        x = Add(name='%s-Add' % feed_forward_name)([xi, x])
        x = LayerNormalization(name='%s-Norm' % feed_forward_name)(x)

    return Model([x_in, s_in], x)


def load_weights_from_checkpoint(model,
                                 checkpoint_file,
                                 config,
                                 keep_words=None):
    """从预训练好的checkpoint中加载权重
    keep_words是词ID组成的list，为精简Embedding层而传入
    """
    loader = partial(tf.train.load_variable, checkpoint_file)
    num_hidden_layers = config['num_hidden_layers']

    if keep_words is None:
        model.get_layer(name='Embedding-Token').set_weights([
            loader('bert/embeddings/word_embeddings'),
        ])
    else:
        model.get_layer(name='Embedding-Token').set_weights([
            loader('bert/embeddings/word_embeddings')[keep_words],
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

    for i in range(num_hidden_layers):
        try:
            model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1))
        except ValueError as e:
            continue
        model.get_layer(
            name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
                loader('bert/encoder/layer_%d/attention/self/query/kernel' % i),
                loader('bert/encoder/layer_%d/attention/self/query/bias' % i),
                loader('bert/encoder/layer_%d/attention/self/key/kernel' % i),
                loader('bert/encoder/layer_%d/attention/self/key/bias' % i),
                loader('bert/encoder/layer_%d/attention/self/value/kernel' % i),
                loader('bert/encoder/layer_%d/attention/self/value/bias' % i),
                loader('bert/encoder/layer_%d/attention/output/dense/kernel' % i),
                loader('bert/encoder/layer_%d/attention/output/dense/bias' % i),
            ])
        model.get_layer(
            name='Encoder-%d-MultiHeadSelfAttention-Norm' %
            (i + 1)).set_weights([
                loader(
                    'bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
                loader(
                    'bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
            ])
        model.get_layer(
            name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
                loader(
                    'bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
                loader(
                    'bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
            ])
        model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
            loader('bert/encoder/layer_%d/intermediate/dense/bias' % i),
            loader('bert/encoder/layer_%d/output/dense/kernel' % i),
            loader('bert/encoder/layer_%d/output/dense/bias' % i),
        ])
        model.get_layer(
            name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
                loader('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
                loader('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
            ])


def load_pretrained_model(config_path,
                          checkpoint_file,
                          seq2seq=False,
                          keep_words=None):
    """根据配置文件和checkpoint文件来加载模型
    """
    config = json.load(open(config_path))
    if keep_words is None:
        vocab_size = config['vocab_size']
    else:
        vocab_size = len(keep_words)
    model = get_bert_model(
        vocab_size=vocab_size,
        max_position_embeddings=config['max_position_embeddings'],
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        hidden_act=config['hidden_act'],
        dropout_rate=0.1,
        seq2seq=seq2seq)
    load_weights_from_checkpoint(model, checkpoint_file, config, keep_words)
    return model
