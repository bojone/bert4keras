#! -*- coding: utf-8 -*-
# 基本测试：中文GPT2模型
# 介绍链接：https://kexue.fm/archives/7292

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout


config_path = '/root/gpt2/config.json'
checkpoint_path = '/root/gpt2/model.ckpt-100000'
dict_path = '/root/gpt2/vocab.txt'

tokenizer = Tokenizer(dict_path,
                      token_start=None,
                      token_end=None,
                      do_lower_case=True)  # 建立分词器

model = build_transformer_model(config_path=config_path,
                                checkpoint_path=checkpoint_path,
                                model='gpt2_ml')  # 建立模型，加载权重


class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids = np.concatenate([inputs[0], output_ids], 1)
        return model.predict(token_ids)[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids], n, topk)  # 基于随机采样
        return [text + tokenizer.decode(ids) for ids in results]


article_completion = ArticleCompletion(start_id=None,
                                       end_id=511,  # 511是中文句号
                                       maxlen=256,
                                       minlen=128)

print(article_completion.generate(u'今天天气不错'))
