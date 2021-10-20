#! -*- coding:utf-8 -*-
# 文本分类多gpu版
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)

import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import json
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from tqdm import tqdm

num_classes = 119
maxlen = 128
batch_size = 32

# BERT base
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data(
    '/root/CLUE-master/baselines/CLUEdataset/iflytek/train.json'
)
valid_data = load_data(
    '/root/CLUE-master/baselines/CLUEdataset/iflytek/dev.json'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            yield [token_ids, segment_ids], [[label]]  # 返回一条样本


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

# 建立单机多卡策略
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():  # 调用该策略

    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=None,
        return_keras_model=False,
    )

    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=bert.initializer,
        name='Probas'
    )(output)

    model = keras.models.Model(bert.model.input, output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(2e-5),
        metrics=['sparse_categorical_accuracy'],
    )
    model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)  # 必须最后才加载预训练权重


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs['sparse_categorical_accuracy']
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    train_dataset = train_generator.to_dataset(
        types=[('float32', 'float32'), ('float32',)],
        shapes=[([None], [None]), ([1],)],  # 配合后面的padded_batch=True，实现自动padding
        names=[('Input-Token', 'Input-Segment'), ('Probas',)],
        padded_batch=True
    )  # 数据要转为tf.data.Dataset格式，names跟输入层/输出层的名字对应

    valid_dataset = valid_generator.to_dataset(
        types=[('float32', 'float32'), ('float32',)],
        shapes=[([None], [None]), ([1],)],  # 配合后面的padded_batch=True，实现自动padding
        names=[('Input-Token', 'Input-Segment'), ('Probas',)],
        padded_batch=True
    )  # 数据要转为tf.data.Dataset格式，names跟输入层/输出层的名字对应

    model.fit(
        train_dataset,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=valid_dataset,
        validation_steps=len(valid_generator),
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/iflytek/test.json', 'iflytek_predict.json')
