#! -*- coding: utf-8- -*-
# 用Seq2Seq做阅读理解构建
# 根据篇章先采样生成答案，然后采样生成问题
# 数据集同 https://github.com/bojone/dgcnn_for_reading_comprehension

import json, os
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.snippets import text_segmentate
from keras.models import Model
from tqdm import tqdm

# 基本参数
max_p_len = 128
max_q_len = 64
max_a_len = 16
batch_size = 32
epochs = 100

# bert配置
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

# 标注数据
webqa_data = json.load(open('/root/qa_datasets/WebQA.json'))
sogou_data = json.load(open('/root/qa_datasets/SogouQA.json'))

# 筛选数据
seps, strips = u'\n。！？!?；;，, ', u'；;，, '
data = []
for d in webqa_data + sogou_data:
    for p in d['passages']:
        if p['answer']:
            for t in text_segmentate(p['passage'], max_p_len - 2, seps, strips):
                if p['answer'] in t:
                    data.append((t, d['question'], p['answer']))

del webqa_data
del sogou_data

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

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]
        """
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (p, q, a) in self.sample(random):
            p_token_ids, _ = tokenizer.encode(p, maxlen=max_p_len + 1)
            a_token_ids, _ = tokenizer.encode(a, maxlen=max_a_len)
            q_token_ids, _ = tokenizer.encode(q, maxlen=max_q_len)
            token_ids = p_token_ids + a_token_ids[1:] + q_token_ids[1:]
            segment_ids = [0] * len(p_token_ids)
            segment_ids += [1] * (len(token_ids) - len(p_token_ids))
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class QuestionAnswerGeneration(AutoRegressiveDecoder):
    """随机生成答案，并且通过beam search来生成问题
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, passage, topk=5):
        token_ids, segment_ids = tokenizer.encode(passage, maxlen=max_p_len)
        a_ids = self.random_sample([token_ids, segment_ids], 1,
                                   topk)[0]  # 基于随机采样
        token_ids += list(a_ids)
        segment_ids += [1] * len(a_ids)
        q_ids = self.beam_search([token_ids, segment_ids],
                                 topk)  # 基于beam search
        return (tokenizer.decode(q_ids), tokenizer.decode(a_ids))


qag = QuestionAnswerGeneration(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=max_q_len
)


def predict_to_file(data, filename, topk=1):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q, a = qag.generate(d[0])
            s = '%s\t%s\t%s\n' % (q, a, d[0])
            f.write(s)
            f.flush()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=1000,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')
    # predict_to_file(valid_data, 'qa.csv')
