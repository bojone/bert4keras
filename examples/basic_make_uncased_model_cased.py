#! -*- coding: utf-8 -*-
# 通过简单修改词表，使得不区分大小写的模型有区分大小写的能力
# 基本思路：将英文单词大写化后添加到词表中，并修改模型Embedding层

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import to_array
import numpy as np

config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = load_vocab(dict_path)
new_token_dict = token_dict.copy()
compound_tokens = []

for t, i in sorted(token_dict.items(), key=lambda s: s[1]):
    # 这里主要考虑两种情况：1、首字母大写；2、整个单词大写。
    # Python2下，新增了5594个token；Python3下，新增了5596个token。
    tokens = []
    if t.isalpha():
        tokens.extend([t[:1].upper() + t[1:], t.upper()])
    elif t[:2] == '##' and t[2:].isalpha():
        tokens.append(t.upper())
    for token in tokens:
        if token not in new_token_dict:
            compound_tokens.append([i])
            new_token_dict[token] = len(new_token_dict)

tokenizer = Tokenizer(new_token_dict, do_lower_case=False)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    compound_tokens=compound_tokens,  # 增加新token，用旧token平均来初始化
)

text = u'Welcome to BEIJING.'
tokens = tokenizer.tokenize(text)
print(tokens)
"""
输出：['[CLS]', u'Welcome', u'to', u'BE', u'##I', u'##JING', u'.', '[SEP]']
"""

token_ids, segment_ids = tokenizer.encode(text)
token_ids, segment_ids = to_array([token_ids], [segment_ids])
print(model.predict([token_ids, segment_ids]))
"""
输出：
[[[-1.4999904e-01  1.9651388e-01 -1.7924258e-01 ...  7.8269649e-01
    2.2241375e-01  1.1325148e-01]
  [-4.5268752e-02  5.5090344e-01  7.4699545e-01 ... -4.7773960e-01
   -1.7562288e-01  4.1265407e-01]
  [ 7.0158571e-02  1.7816302e-01  3.6949167e-01 ...  9.6258509e-01
   -8.4678203e-01  6.3776302e-01]
  ...
  [ 9.3637377e-01  3.0232478e-02  8.1411439e-01 ...  7.9186147e-01
    7.5704646e-01 -8.3475001e-04]
  [ 2.3699696e-01  2.9953337e-01  8.1962071e-02 ... -1.3776925e-01
    3.8681498e-01  3.2553676e-01]
  [ 1.9728680e-01  7.7782705e-02  5.2951699e-01 ...  8.9622810e-02
   -2.3932748e-02  6.9600858e-02]]]
"""
