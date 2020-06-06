#! -*- coding:utf-8 -*-
# 通过虚拟对抗训练进行半监督学习
# use_vat=True比use_vat=False约有1%的提升
# 数据集：情感分析数据集
# 博客：https://kexue.fm/archives/7466

import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
from keras.utils import to_categorical
from tqdm import tqdm

# 配置信息
num_classes = 2
maxlen = 128
batch_size = 32
train_frac = 0.01  # 标注数据的比例
use_vat = True  # 可以比较True/False的效果

# BERT base
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data('datasets/sentiment/sentiment.train.data')
valid_data = load_data('datasets/sentiment/sentiment.valid.data')
test_data = load_data('datasets/sentiment/sentiment.test.data')

# 模拟标注和非标注数据
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 0) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = to_categorical(batch_labels, num_classes)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

# 用于正常训练的模型
model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='kld',
    optimizer=Adam(2e-5),
    metrics=['categorical_accuracy'],
)

# 用于虚拟对抗训练的模型
model_vat = keras.models.Model(bert.model.input, output)
model_vat.compile(
    loss='kld',
    optimizer=Adam(1e-5),
    metrics=['categorical_accuracy'],
)


def virtual_adversarial_training(
    model, embedding_name, epsilon=1, xi=10, iters=1
):
    """给模型添加虚拟对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    model_outputs = K.function(
        inputs=inputs,
        outputs=model.outputs,
        name='model_outputs',
    )  # 模型输出函数
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 模型梯度函数

    def l2_normalize(x):
        return x / (np.sqrt((x**2).sum()) + 1e-8)

    def train_function(inputs):  # 重新定义训练函数
        outputs = model_outputs(inputs)
        inputs = inputs[:2] + outputs + inputs[3:]
        delta1, delta2 = 0.0, np.random.randn(*K.int_shape(embeddings))
        for _ in range(iters):  # 迭代求扰动
            delta2 = xi * l2_normalize(delta2)
            K.set_value(embeddings, K.eval(embeddings) - delta1 + delta2)
            delta1 = delta2
            delta2 = embedding_gradients(inputs)[0]  # Embedding梯度
        delta2 = epsilon * l2_normalize(delta2)
        K.set_value(embeddings, K.eval(embeddings) - delta1 + delta2)
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta2)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
virtual_adversarial_training(model_vat, 'Embedding-Token')


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.
        self.data = data_generator(unlabeled_data, batch_size).forfit()

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

    def on_batch_end(self, batch, logs=None):
        if use_vat:
            dx, dy = next(self.data)
            model_vat.train_on_batch(dx, dy)


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=30,
        epochs=100,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
