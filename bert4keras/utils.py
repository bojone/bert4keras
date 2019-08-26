#! -*- coding: utf-8 -*-

import tensorflow as tf
from functools import partial


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


class SimpleTokenizer:
    """简单的分词器，直接将文本分割为单字符序列
    """
    def __init__(self, token_dict):
        """初始化词典
        """
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}

    def _is_space(c):
        """判断是否为空格
        """
        return c == ' ' or c == '\n' or c == '\r' or c == '\t' or \
            unicodedata.category(c) == 'Zs'

    def tokenize(self, text, add_cls=True, add_sep=True):
        """按字分割
        """
        R = []
        if add_cls:
            R.append('[CLS]')
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        if add_sep:
            R.append('[SEP]')
        return R

    def encode(self, first, second=None):
        """输出文本对应token id和segment id
        """
        token_ids, segment_ids = [], []
        token_ids.extend([self._token_dict[c] for c in self.tokenize(first)])
        segment_ids.extend([0] * (len(first) + 2))
        if second is not None:
            token_ids.extend([
                self._token_dict[c]
                for c in self.tokenize(second, add_cls=False)
            ])
            segment_ids.extend([1] * (len(second) + 1))
        return token_ids, segment_ids
