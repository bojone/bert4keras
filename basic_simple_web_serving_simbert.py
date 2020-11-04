#! -*- coding: utf-8 -*-
# 利用自带的接口，将SimBERT的同义句生成搭建成Web服务。
# 基于bottlepy简单封装，仅作为临时测试使用，不保证性能。
# 目前仅保证支持 Tensorflow 1.x + Keras <= 2.3.1。
# 具体用法请看 https://github.com/bojone/bert4keras/blob/8ffb46a16a79f87aa8cdf045df7994036b4be47d/bert4keras/snippets.py#L580

import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
from bert4keras.snippets import WebServing

maxlen = 32

# bert配置
config_path = '/root/kg/bert/chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_simbert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n,
                                        topk)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def gen_synonyms(text, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    """
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


if __name__ == '__main__':

    arguments = {'text': (None, True), 'n': (int, False), 'k': (int, False)}
    web = WebServing(port=8864)
    web.route('/gen_synonyms', gen_synonyms, arguments)
    web.start()
    # 现在可以测试访问 https://127.0.0.1:8864/gen_synonyms?text=苹果多少钱一斤
