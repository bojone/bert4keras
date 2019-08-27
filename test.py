#! -*- coding: utf-8 -*-
# 测试代码可用性

import codecs
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
import numpy as np


config_path = '../../kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = load_vocab(dict_path) # 读取词典
tokenizer = SimpleTokenizer(token_dict) # 建立分词器
model = load_pretrained_model(config_path, checkpoint_path) # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
