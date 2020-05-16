#! -*- coding: utf-8 -*-
# bert做conditional language model任务
# 按类随机生成文本，这个demo的类别是情感极性（正／负）
# 请参考：https://kexue.fm/archives/7124

from __future__ import print_function
import re
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import text_segmentate
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.snippets import uniout  # 打印中文
from keras.layers import Input, Embedding, Reshape
from keras.models import Model

# 模型配置
maxlen = 128
batch_size = 32
num_classes = 2
epochs = 20

# bert配置
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def load_data(filenames):
    """加载数据，并尽量划分为不超过maxlen的句子
    """
    D = []
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for l in f:
                text, label = l.strip().split('\t')
                for t in text_segmentate(text, maxlen - 2, seps, strips):
                    D.append((t, int(label)))
    return D


# 加载数据集
data = load_data([
    'datasets/sentiment/sentiment.train.data',
    'datasets/sentiment/sentiment.valid.data',
    'datasets/sentiment/sentiment.test.data',
])


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids, batch_labels], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        if mask[1] is None:
            y_mask = 1.0
        else:
            y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


c_in = Input(shape=(1,))
c = Embedding(2, 128)(c_in)
c = Reshape((128,))(c)

# Bert模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='lm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    layer_norm_cond=c,
    additional_input_layers=c_in,
)

output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class RandomSentiment(AutoRegressiveDecoder):
    """根据情感标签（0:负，1:正）随机生成一批句子
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids = output_ids
        segment_ids = np.zeros_like(token_ids)
        return model.predict([token_ids, segment_ids, inputs[0]])[:, -1]

    def generate(self, label, n=1, topk=5):
        results = self.random_sample([[label]], n, topk)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in results]


random_sentiment = RandomSentiment(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=maxlen
)


def just_show():
    print(u'正面采样:')
    print(random_sentiment.generate(1, 5, 5), '\n')
    print(u'负面采样:')
    print(random_sentiment.generate(0, 5, 5), '\n')


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        # 演示效果
        just_show()


if __name__ == '__main__':

    evaluator = Evaluate()
    train_generator = data_generator(data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')
"""
正面采样:
[
    u'外观时尚、漂亮、性价比高。',
    u'外观漂亮，配置均衡，比较满意，性价比高，外观漂亮，性能较高。',
    u'我是在大学的时候看到这本书的，所以一直在买。书中的作者是林静蕾，她用自己的口吻写出了一个孩子成长中的心路历程，让我看到了她们成长中的不同之处，以及她们成长过程中的不同境界。让我很欣赏！',
    u'我想这是一本能够告诉读者什么是坏的，而不是教你怎样说话，告诉我什么是错。这里我推荐了《我要讲故事》，这本书是我很喜欢的一本书，我认为它的理由很多，但是，我相信我。如果你从中得到一些改进，或者你已经有了一个明智的决定。',
    u'我们一家五口住的是标间，大床房，大床的床很舒服；而我们在携程网上订了两套大床房，这个酒店的价格还是比较合理的；但是房间的隔音效果不太理想，有点响的声音；酒店门口的地铁在施工中，不方便；但是酒店的门口的出租车不知道是哪个车的，打车不是很方便；酒店外面的停'
]

负面采样:
[
    u'不知道是不是因为电池不太好，不是我不喜欢。',
    u'看了评论才买的. 结果发现不是那么便宜, 价格也不便宜.',
    u'1、外壳不容易沾手印，不容易洗洗2、屏幕有点旧， 不能下载铃声',
    u'我是7月6日订购了《杜拉拉升职记》并已通过银行付款，为什么订单下了两周多至今还未到货？是收货时间太快了，可能就这么过去了吧？',
    u'这本书我是在网上先看了一遍，后来我再看了一遍。感觉作者的文笔实在太烂了，特别是在写他的博客时特别别扭，写得很不专业，特别是他写股票时那个情绪调节的小男孩，简直就是自作聪明的样子，简直就是自作聪明的一种表现！'
]
"""
