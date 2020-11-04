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

tokenizer = Tokenizer(
    dict_path, token_start=None, token_end=None, do_lower_case=True
)  # 建立分词器

model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, model='gpt2_ml'
)  # 建立模型，加载权重


class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = np.concatenate([inputs[0], output_ids], 1)
        return model.predict(token_ids)[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids], n, topk)  # 基于随机采样
        return [text + tokenizer.decode(ids) for ids in results]


article_completion = ArticleCompletion(
    start_id=None,
    end_id=511,  # 511是中文句号
    maxlen=256,
    minlen=128
)

print(article_completion.generate(u'今天天气不错'))
"""
部分结果：

>>> article_completion.generate(u'今天天气不错')
[u'今天天气不错，可以去跑步。昨晚看了一个关于跑步的纪录片，里面的女主讲述的是一个女孩子的成长，很励志，也很美丽。我也想跑，但是我不知道跑步要穿运动鞋，所以就买了一双运动鞋。这个纪录片是关于运动鞋的，有一 集讲了一个女孩子，从小学开始就没有穿过运动鞋，到了高中才开始尝试跑步。']

>>> article_completion.generate(u'双十一')
[u'双十一马上就要到了！你还在为双11的物流配送而担心吗？你还在为没时间去仓库取货而发愁吗？你还在为不知道怎么买到便宜货而发愁吗？你还在为买不到心仪的产品而懊恼吗？那么，双十一就来了！今天小编带你来看看这些 快递，都是怎么送货的！1. 物流配送快递公司的配送，主要是由快递公司负责，快递公司负责派件，物流服务。']

>>> article_completion.generate(u'科学空间')
[u'科学空间站科学空间站（英文：science space station），是中华人民共和国的一个空间站。该空间站是中国科学院大连物理研究所研制，主要研发和使用中国科学院大连物理研究所的核能动力空间站。科学空间站位于北京市海淀区，距离地面393米，总建筑面积约为1万平方米，总投资约为5亿元人民币。科学空间站于2018年12月26日开始动工，2021年6月建成并投入使用。']
"""
