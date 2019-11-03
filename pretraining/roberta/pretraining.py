#! -*- coding: utf-8 -*-
# RoBERTa预训练脚本，多GPU版

import os, re
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import tensorflow as tf
from data_utils import TrainingDataset
from bert4keras.bert import build_bert_model
from bert4keras.backend import keras, K
from bert4keras.backend import piecewise_linear
from bert4keras.train import add_weight_decay_into
from tensorflow.python.framework import ops


# 自定义配置
corpus_path = '../../test.tfrecord'
sequence_length = 256
batch_size = 64
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
learning_rate = 5e-5
weight_decay_rate = 0.01
num_warmup_steps = 10000
num_train_steps = 1000000
steps_per_epoch = 2000
epochs = num_train_steps // steps_per_epoch
exclude_from_weight_decay = ['Norm', 'bias']


# 准备一些变量
Input = keras.layers.Input
Lambda = keras.layers.Lambda
Model = keras.models.Model
sparse_categorical_accuracy = keras.metrics.sparse_categorical_accuracy
ModelCheckpoint = keras.callbacks.ModelCheckpoint


# 读取数据集，构建数据张量
dataset = TrainingDataset.load_tfrecord(
    record_names=corpus_path,
    sequence_length=sequence_length,
    batch_size=batch_size,
)


# 构建优化器

LearningRateSchedule = keras.optimizers.schedules.LearningRateSchedule


class PiecewiseLinear(LearningRateSchedule):
    """为tf.keras的OptimizerV2所写的分段线性学习率
    """
    def __init__(self, schedule, name=None):
        super(PiecewiseLinear, self).__init__()
        self.schedule = {int(i): j for i, j in schedule.items()}
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "PiecewiseLinear") as name:
            return piecewise_linear(step, self.schedule)

    def get_config(self):
        return {'schedule': self.schedule, 'name': self.name}


lr_schedule = {num_warmup_steps: learning_rate, num_train_steps: 0.}
optimizer = keras.optimizers.Adam(learning_rate=PiecewiseLinear(lr_schedule))


# 构建模型

def build_train_bert_model():

    # 基本模型
    bert_model = build_bert_model(config_path, with_mlm=True)

    token_ids = Input(shape=(None, ), dtype='int32')  # 原始token_ids
    mask_ids = Input(shape=(None, ), dtype='int32')  # 被mask的token的标记

    # RoBERTa模式直接使用全零segment_ids
    segment_ids = Lambda(lambda x: K.zeros_like(x, dtype='int32'), )(token_ids)

    # 是否被mask的标记
    is_masked = Lambda(lambda x: K.not_equal(x, 0))(mask_ids)

    # 将指定token替换为mask
    def random_mask(inputs):
        token_ids, mask_ids, is_masked = inputs
        return K.switch(is_masked, mask_ids - 1, token_ids)

    masked_token_ids = Lambda(random_mask)([token_ids, mask_ids, is_masked])

    # 计算概率
    proba = bert_model([masked_token_ids, segment_ids])

    # 构建训练模型
    train_model = Model([token_ids, mask_ids], proba)

    # 提取被mask部分，然后构建loss
    indices = tf.where(is_masked)
    y_pred = tf.gather_nd(proba, indices)
    y_true = tf.gather_nd(token_ids, indices)
    mlm_loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    mlm_loss = K.mean(mlm_loss)
    train_model.add_loss(mlm_loss)

    # 计算accuracy
    mlm_acc = sparse_categorical_accuracy(K.cast(y_true, K.floatx()), y_pred)
    mlm_acc = K.mean(mlm_acc)
    train_model.add_metric(mlm_acc, name='accuracy', aggregation='mean')

    # 添加权重衰减
    add_weight_decay_into(train_model, weight_decay_rate,
                          exclude_from_weight_decay)

    # 模型定型
    train_model.compile(optimizer=optimizer)
    return train_model


# 多GPU模式
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    train_model = build_train_bert_model()

# 模型训练
train_model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs)
