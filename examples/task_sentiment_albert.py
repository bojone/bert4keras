#! -*- coding:utf-8 -*-
# 情感分析类似，加载albert_zh权重(https://github.com/brightmart/albert_zh)

import json
import numpy as np
import pandas as pd
from random import choice
import re, os
import codecs
from bert4keras.backend import set_gelu
from bert4keras.utils import Tokenizer, load_vocab
from bert4keras.bert import build_bert_model
from bert4keras.train import PiecewiseLinearLearningRate
set_gelu('tanh') # 切换gelu版本


maxlen = 100
config_path = '/root/kg/bert/albert_base_zh/bert_config.json'
checkpoint_path = '/root/kg/bert/albert_base_zh/bert_model.ckpt'
dict_path = '/root/kg/bert/albert_base_zh/vocab.txt'


neg = pd.read_excel('datasets/neg.xls', header=None)
pos = pd.read_excel('datasets/pos.xls', header=None)
data, tokens = [], {}

_token_dict = load_vocab(dict_path) # 读取词典
_tokenizer = Tokenizer(_token_dict) # 建立临时分词器

for d in neg[0]:
    data.append((d, 0))
    for t in _tokenizer.tokenize(d):
        tokens[t] = tokens.get(t, 0) + 1

for d in pos[0]:
    data.append((d, 1))
    for t in _tokenizer.tokenize(d):
        tokens[t] = tokens.get(t, 0) + 1

tokens = {i: j for i, j in tokens.items() if j >= 4}
token_dict, keep_words = {}, [] # keep_words是在bert中保留的字表

for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_dict[t] = len(token_dict)
    keep_words.append(_token_dict[t])

for t in tokens:
    if t in _token_dict and t not in token_dict:
        token_dict[t] = len(token_dict)
        keep_words.append(_token_dict[t])

tokenizer = Tokenizer(token_dict) # 建立分词器


if not os.path.exists('./random_order.json'):
    random_order = range(len(data))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('./random_order.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('./random_order.json'))


# 按照9:1的比例划分训练集和验证集
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam


model = build_bert_model(
    config_path,
    checkpoint_path,
    keep_words=keep_words, # 只保留keep_words中的字，精简原字表
    albert=True
)

output = Lambda(lambda x: x[:, 0])(model.output)
output = Dense(1, activation='sigmoid')(output)
model = Model(model.input, output)

model.compile(
    loss='binary_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=PiecewiseLinearLearningRate(Adam(1e-4), {1000: 1, 2000: 0.1}),
    metrics=['accuracy']
)
model.summary()


train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=10,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)
