#! -*- coding: utf-8 -*-

from layers import *


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


def get_bert_encoder_from_config(config):
    """根据预先设定的配置字典来构建模型
    """
    model = get_bert_encoder(
        vocab_size=config['vocab_size'],
        max_position_embeddings=config['max_position_embeddings'],
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        hidden_act=config['hidden_act'],
        dropout_rate=0.1)
    return model
