#! -*- coding: utf-8 -*-
# RoBERTa预训练脚本，多GPU版/TPU版本

import os, re
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import tensorflow as tf
from data_utils import TrainingDataset
from bert4keras.bert import build_bert_model
from bert4keras.backend import keras, K
from bert4keras.backend import piecewise_linear
from bert4keras.train import LAMB, add_weight_decay_into
from tensorflow.python.framework import ops


# 语料路径和模型保存路径
# 如果是TPU训练，那么语料必须存放在Google Cloud Storage上面，
# 路径必须以gs://开通；如果是GPU训练，改为普通路径即可。
corpus_path = 'gs://xxxx/bert4keras/corpus.tfrecord'
saved_model_path = 'gs://xxxx/bert4keras/saved_model/bert_model.ckpt'

# 其他配置
sequence_length = 512
batch_size = 512
config_path = '/home/spaces_ac_cn/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/spaces_ac_cn/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt' # 如果从零训练，就设为None
learning_rate = 5e-5
weight_decay_rate = 0.01
num_warmup_steps = 10000
num_train_steps = 1000000
steps_per_epoch = 2000
epochs = num_train_steps // steps_per_epoch
exclude_from_weight_decay = ['Norm', 'bias']
tpu_address = 'grpc://xxx.xxx.xxx.xxx:8470' # 如果用多GPU跑，直接设为None
which_optimizer = 'adam'  # adam 或 lamb，均自带weight decay
lr_schedule = {num_warmup_steps: learning_rate, num_train_steps: 0.}

# 准备变量
Input = keras.layers.Input
Lambda = keras.layers.Lambda
Adam = keras.optimizers.Adam
Model = keras.models.Model
ModelCheckpoint = keras.callbacks.ModelCheckpoint
CSVLogger = keras.callbacks.CSVLogger

# 读取数据集，构建数据张量
dataset = TrainingDataset.load_tfrecord(
    record_names=corpus_path,
    sequence_length=sequence_length,
    batch_size=batch_size,
)


class PiecewiseLinear(keras.optimizers.schedules.LearningRateSchedule):
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


def build_train_bert_model():
    """构建训练模型，通用于TPU/GPU
    注意全程要用keras标准的层写法，一些比较灵活的“移花接木”式的
    写法可能会在TPU上训练失败。此外，要注意的是TPU并非支持所有
    tensorflow算子，尤其不支持动态（变长）算子，因此编写相应运算
    时要格外留意。
    """
    bert = build_bert_model(config_path, with_mlm='linear', return_keras_model=False)
    bert_model = bert.model
    proba = bert_model.output

    # 辅助输入
    token_ids = Input(shape=(None, ), dtype='int64', name='token_ids') # 目标id
    is_masked = Input(shape=(None, ), dtype='bool', name='is_masked') # mask标记

    def mlm_loss(inputs):
        """计算loss的函数，需要封装为一个层
        """
        y_true, y_pred, is_masked = inputs
        is_masked = K.cast(is_masked, K.floatx())
        loss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss = K.sum(loss * is_masked) / (K.sum(is_masked) + K.epsilon())
        return loss

    def mlm_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred, is_masked = inputs
        is_masked = K.cast(is_masked, K.floatx())
        y_true = K.cast(y_true, K.floatx())
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * is_masked) / (K.sum(is_masked) + K.epsilon())
        return acc

    loss = Lambda(mlm_loss, name='mlm_loss')([token_ids, proba, is_masked])
    acc = Lambda(mlm_acc, name='mlm_acc')([token_ids, proba, is_masked])

    train_model = Model(bert_model.inputs + [token_ids, is_masked], [loss, acc])

    # 优化器
    if which_optimizer == 'adam':
        optimizer = Adam(learning_rate=PiecewiseLinear(lr_schedule))
        learning_rate = optimizer._decayed_lr(tf.float32)
        # 添加权重衰减
        add_weight_decay_into(bert_model, weight_decay_rate * learning_rate,
                              exclude_from_weight_decay)
    else:
        optimizer = LAMB(learning_rate=PiecewiseLinear(lr_schedule),
                         weight_decay_rate=weight_decay_rate,
                         exclude_from_weight_decay=exclude_from_weight_decay)

    # 模型定型
    train_model.compile(
        loss={
            'mlm_loss': lambda y_true, y_pred: y_pred,
            'mlm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
        },
        optimizer=optimizer,
    )

    # 如果传入权重，则加载。注：须在此处加载，才保证不报错。
    if checkpoint_path is not None:
        bert.load_weights_from_checkpoint(checkpoint_path)

    return train_model


if tpu_address is None:
    # 单机多卡模式（多机多卡也类似，但需要硬软件配合，请参考https://tf.wiki）
    strategy = tf.distribute.MirroredStrategy()
else:
    # TPU模式
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    tf.config.experimental_connect_to_host(resolver.master())
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
    train_model = build_train_bert_model()
    train_model.summary()

# 模型回调
checkpoint = ModelCheckpoint(
    filepath=saved_model_path,
    monitor='mlm_loss_loss',
    save_weights_only=True,
    save_best_only=True,
)
csv_logger = CSVLogger('training.log') # 记录训练日志

# 模型训练
train_model.fit(
    dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[checkpoint, csv_logger],
)
