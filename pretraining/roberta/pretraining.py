#! -*- coding: utf-8 -*-
# RoBERTa预训练脚本，单GPU版

from data_utils import TrainingDataset
from bert4keras.bert import build_bert_model
from bert4keras.backend import keras, K
from bert4keras.train import PiecewiseLinearLearningRate
from bert4keras.train import GradientAccumulation
import tensorflow as tf


# 自定义配置
dataset_path = '../../test.tfrecord'
padding_length = 256
batch_size = 32
token_mask_id = 103
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
optimizer = keras.optimizers.Adam(1e-5)
steps_per_epoch = 2000
epochs = 1000


# 准备一些变量
Input = keras.layers.Input
Lambda = keras.layers.Lambda
Model = keras.models.Model
sparse_categorical_accuracy = keras.metrics.sparse_categorical_accuracy


# 读取数据集，构建数据张量
dataset = TrainingDataset.load_tfrecord(
    record_names=dataset_path,
    padding_length=padding_length,
    batch_size=batch_size,
)


# 基本模型
bert_model = build_bert_model(config_path, with_mlm=True)

token_ids = Input(tensor=dataset[0])  # 原始token_ids
mask_ids = Input(tensor=dataset[1])  # 被mask的token的标记

# RoBERTa模式直接使用全零segment_ids
segment_ids = Lambda(lambda x: K.zeros_like(x, dtype='int32'), )(token_ids)

# 将指定token替换为mask
def random_mask(inputs):
    token_ids, mask_ids = inputs
    cond = K.equal(mask_ids, 1)
    return K.switch(cond, mask_ids, token_ids)

masked_token_ids = Lambda(random_mask)([token_ids, mask_ids])

# 计算概率
proba = bert_model([masked_token_ids, segment_ids])

# 构建训练模型
train_model = Model([token_ids, mask_ids], proba)

# 提取被mask部分，然后构建loss
indices = tf.where(tf.equal(mask_ids, 1))
y_pred = tf.gather_nd(proba, indices)
y_true = tf.gather_nd(token_ids, indices)
mlm_loss = K.sparse_categorical_crossentropy(y_true, y_pred)
mlm_loss = K.mean(mlm_loss)
train_model.add_loss(mlm_loss)

# 计算accuracy
mlm_acc = sparse_categorical_accuracy(K.cast(y_true, K.floatx()), y_pred)
mlm_acc = K.mean(mlm_acc)
train_model.add_metric(mlm_acc, name='accuracy', aggregation='mean')

# 模型定型
train_model.compile(optimizer=optimizer)

# 模型训练
train_model.fit(steps_per_epoch=steps_per_epoch, epochs=epochs)
