#! -*- coding:utf-8 -*-
# 情感分析例子，加载albert_zh权重(https://github.com/brightmart/albert_zh)

import json
import numpy as np
from random import choice
import re, os, codecs
from bert4keras.backend import set_gelu
from bert4keras.utils import Tokenizer, load_vocab
from bert4keras.bert import build_bert_model
from bert4keras.train import PiecewiseLinearLearningRate
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback

set_gelu('tanh') # 切换gelu版本


maxlen = 128
config_path = '/root/kg/bert/albert_small_zh_google/albert_config.json'
checkpoint_path = '/root/kg/bert/albert_small_zh_google/albert_model.ckpt'
dict_path = '/root/kg/bert/albert_small_zh_google/vocab.txt'


def load_data(filename):
    D = []
    with codecs.open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


train_data = load_data('datasets/sentiment/sentiment.train.data')
valid_data = load_data('datasets/sentiment/sentiment.valid.data')
test_data = load_data('datasets/sentiment/sentiment.test.data')
tokenizer = Tokenizer(dict_path) # 建立分词器


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
        for x in X
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
        idxs = list(range(len(self.data)))
        np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text, label = self.data[i]
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = seq_padding(batch_token_ids)
                batch_segment_ids = seq_padding(batch_segment_ids)
                batch_labels = seq_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    def forfit(self):
        while True:
            for d in self.__iter__():
                yield d


model = build_bert_model(config_path, checkpoint_path, albert=True) # 加载预训练模型
output = Lambda(lambda x: x[:, 0])(model.output) # 取出[CLS]对应的向量
output = Dense(1, activation='sigmoid')(output) # 接一个二分类器
model = Model(model.input, output) # 建立模型

model.compile(
    loss='binary_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=PiecewiseLinearLearningRate(Adam(1e-4), {1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)
model.summary()


train_generator = data_generator(train_data)
valid_generator = data_generator(valid_data)
test_generator = data_generator(test_data)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true) >= 0.5
        y_true = y_true >= 0.5
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(Callback):
    def __init__(self):
        self.best_acc = 0.
    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(valid_generator)
        if acc > self.best_acc:
            self.best_acc = acc
            model.save_weights('best_model.weights')
        print(u'acc: %05f, best_acc: %05f\n' % (acc, self.best_acc))


evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    callbacks=[evaluator])

model.load_weights('best_model.weights')
print(u'test acc: %05f\n' % (evaluate(test_generator)))
