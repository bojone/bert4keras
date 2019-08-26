#! -*- coding: utf-8 -*-
# 测试代码可用性

import codecs, json
from bert4keras.bert import get_bert_encoder_from_config
from bert4keras.utils import SimpleTokenizer, load_weights_from_checkpoint
import numpy as np


config_path = '../../kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


token_dict = {}
with codecs.open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = SimpleTokenizer(token_dict) # 建立分词器


config = json.load(open(config_path)) # 读取配置文件
model = get_bert_encoder_from_config(config) # 建立模型
load_weights_from_checkpoint(model, checkpoint_path, config) # 加载权重


token_ids, segment_ids = tokenizer.encode(u'语言模型')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
