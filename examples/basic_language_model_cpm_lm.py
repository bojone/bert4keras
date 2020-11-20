#! -*- coding: utf-8 -*-
# 基本测试：清华开源的中文GPT2模型（26亿参数）
# 项目链接：https://github.com/TsinghuaAI/CPM-Generate
# 博客介绍：https://kexue.fm/archives/7912

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import SpTokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout
import jieba
jieba.initialize()

# 模型路径
config_path = '/root/kg/bert/CPM_LM_2.6B_TF/config.json'
checkpoint_path = '/root/kg/bert/CPM_LM_2.6B_TF/model.ckpt'
spm_path = '/root/kg/bert/CPM_LM_2.6B_TF/chinese_vocab.model'


def pre_tokenize(text):
    """分词前处理函数
    """
    return [
        w.replace(' ', u'\u2582').replace('\n', u'\u2583')
        for w in jieba.cut(text, cut_all=False)
    ]


tokenizer = SpTokenizer(
    spm_path,
    token_start=None,
    token_end=None,
    pre_tokenize=pre_tokenize,
    token_translate={u'\u2583': '<cls>'}
)  # 建立分词器

model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, model='gpt2'
)  # 建立模型，加载权重


class TextExpansion(AutoRegressiveDecoder):
    """基于随机采样的文本续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = np.concatenate([inputs[0], output_ids], 1)
        return model.predict(token_ids)[:, -1]

    def generate(self, text, n=1, topp=0.95, temperature=1):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids],
                                     n,
                                     topp=topp,
                                     temperature=temperature)  # 基于随机采样
        results = [token_ids + [int(i) for i in ids] for ids in results]
        texts = [tokenizer.decode(ids) for ids in results]
        return [self.post_replace(text) for text in texts]

    def post_replace(self, text):
        for s, t in [(' ', ''), (u'\u2582', ' '), (u'\u2583', '\n')]:
            text = text.replace(s, t)
        return text


text_expansion = TextExpansion(
    start_id=None,
    end_id=3,  # 3是<cls>，也是换行符
    maxlen=16,
)

print(text_expansion.generate(u'美国的首都是华盛顿\n法国的首都是巴黎\n日本的首都是东京\n中国的首都是', 1)[0])
print(text_expansion.generate(u'书写英文：\n狗 dog\n猫 cat\n鸟 ', 1)[0])
"""
美国的首都是华盛顿
法国的首都是巴黎
日本的首都是东京
中国的首都是北京

书写英文:
狗 dog
猫 cat
鸟 bird
"""
