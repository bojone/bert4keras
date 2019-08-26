#! -*- coding: utf-8 -*-

from layers import *
from functools import partial
import json


def get_bert_encoder(vocab_size, max_position_embeddings, hidden_size,
                     num_hidden_layers, num_attention_heads, intermediate_size,
                     hidden_act, dropout_rate):
    """加载跟Bert一样结构的Transformer-based Encoder模型
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
                               name=attention_name)([x, x, x, mask, mask])
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


def load_weights_from_checkpoint(model, checkpoint_file, config):
    """从预训练好的checkpoint中加载权重
    """
    loader = partial(tf.train.load_variable, checkpoint_file)
    num_hidden_layers = config['num_hidden_layers']

    model.get_layer(name='Embedding-Token').set_weights([
        loader('bert/embeddings/word_embeddings'),
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


def load_pretrained_model(config_path, checkpoint_file):
    """根据配置文件和checkpoint文件来构建模型
    """
    config = json.load(open(config_path))
    model = get_bert_encoder(
        vocab_size=config['vocab_size'],
        max_position_embeddings=config['max_position_embeddings'],
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        hidden_act=config['hidden_act'],
        dropout_rate=0.1)
    load_weights_from_checkpoint(model, checkpoint_file, config)
    return model
