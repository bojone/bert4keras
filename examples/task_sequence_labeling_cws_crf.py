#! -*- coding: utf-8 -*-
# 用CRF做中文分词（CWS, Chinese Word Segmentation）
# 数据集 http://sighan.cs.uchicago.edu/bakeoff2005/
# 最后测试集的F1约为96.1%

import re, os, json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

maxlen = 256
epochs = 10
num_labels = 4
batch_size = 32
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1  # 必要时扩大CRF层的学习率

# bert配置
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            D.append(re.split(' +', l.strip()))
    return D


# 标注数据
data = load_data('/root/icwb2-data/training/pku_training.utf8')

# 保存一个随机序（供划分valid用）
if not os.path.exists('../random_order.json'):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    json.dump(random_order, open('../random_order.json', 'w'), indent=4)
else:
    random_order = json.load(open('../random_order.json'))

# 划分valid
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        """标签含义
        0: 单字词； 1: 多字词首字； 2: 多字词中间； 3: 多字词末字
        """
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if len(w_token_ids) == 1:
                        labels += [0]
                    else:
                        labels += [1] + [2] * (len(w_token_ids) - 2) + [3]
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


"""
后面的代码使用的是bert类型的模型，如果你用的是albert，那么前几行请改为：

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='albert',
)

output_layer = 'Transformer-FeedForward-Norm'
output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
"""

model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learing_rate),
    metrics=[CRF.sparse_accuracy]
)


class WordSegmenter(ViterbiDecoder):
    """基本分词器
    """
    def tokenize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        nodes = model.predict([[token_ids], [segment_ids]])[0]
        labels = self.decode(nodes)
        words = []
        for i, label in enumerate(labels[1:-1]):
            if label < 2:
                words.append([i + 1])
            else:
                words[-1].append(i + 1)
        return [text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1] for w in words]


segmenter = WordSegmenter(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def simple_evaluate(data):
    """简单的评测
    该评测指标不等价于官方的评测指标，但基本呈正相关关系，
    可以用来快速筛选模型。
    """
    total, right = 0., 0.
    for w_true in tqdm(data):
        w_pred = segmenter.tokenize(''.join(w_true))
        w_pred = set(w_pred)
        w_true = set(w_true)
        total += len(w_true)
        right += len(w_true & w_pred)
    return right / total


def predict_to_file(in_file, out_file):
    """预测结果到文件，便于用官方脚本评测
    使用示例：
    predict_to_file('/root/icwb2-data/testing/pku_test.utf8', 'myresult.txt')
    官方评测代码示例：
    data_dir="/root/icwb2-data"
    $data_dir/scripts/score $data_dir/gold/pku_training_words.utf8 $data_dir/gold/pku_test_gold.utf8 myresult.txt > myscore.txt
    （执行完毕后查看myscore.txt的内容末尾）
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file, encoding='utf-8') as fr:
        for l in tqdm(fr):
            l = l.strip()
            if l:
                l = ' '.join(word_segment(l))
            fw.write(l + '\n')
    fw.close()


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        segmenter.trans = trans
        print(segmenter.trans)
        acc = simple_evaluate(valid_data)
        # 保存最优
        if acc >= self.best_val_acc:
            self.best_val_acc = acc
            model.save_weights('./best_model.weights')
        print('acc: %.5f, best acc: %.5f' % (acc, self.best_val_acc))


if __name__ == '__main__':

    evaluator = Evaluate()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')
