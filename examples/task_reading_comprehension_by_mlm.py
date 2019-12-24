#! -*- coding: utf-8 -*-
# 用MLM的方式阅读理解任务
# 数据集和评测同 https://github.com/bojone/dgcnn_for_reading_comprehension
# 4个epoch后在valid上能达到0.78+的分数
# (Accuracy=0.7361642181525457  F1=0.8250514654830805   Final=0.7806078418178131)

import json, os
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K, search_layer
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
import codecs, re
from tqdm import tqdm


max_p_len = 256
max_q_len = 64
max_a_len = 32
batch_size = 32
epochs = 8

# bert配置
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

# 标注数据
webqa_data = json.load(open('/root/qa_datasets/WebQA.json'))
sogou_data = json.load(open('/root/qa_datasets/SogouQA.json'))

# 保存一个随机序（供划分valid用）
if not os.path.exists('../random_order.json'):
    random_order = range(len(sogou_data))
    np.random.shuffle(random_order)
    json.dump(random_order, open('../random_order.json', 'w'), indent=4)
else:
    random_order = json.load(open('../random_order.json'))

# 划分valid
train_data = [sogou_data[j] for i, j in enumerate(random_order) if i % 3 != 0]
valid_data = [sogou_data[j] for i, j in enumerate(random_order) if i % 3 == 0]
train_data.extend(train_data)
train_data.extend(webqa_data)  # 将SogouQA和WebQA按2:1的比例混合

# 加载并精简词表，建立分词器
_token_dict = load_vocab(dict_path)  # 读取词典
token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_dict[t] = len(token_dict)
    keep_words.append(_token_dict[t])

for t, _ in sorted(_token_dict.items(), key=lambda s: s[1]):
    if t not in token_dict:
        if len(t) == 3 and (Tokenizer._is_cjk_character(t[-1])
                            or Tokenizer._is_punctuation(t[-1])):
            continue
        token_dict[t] = len(token_dict)
        keep_words.append(_token_dict[t])

tokenizer = Tokenizer(token_dict, do_lower_case=True)  # 建立分词器


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        """单条样本格式为
        输入：[CLS]篇章[SEP]问题[SEP][MASK][MASK][SEP]
        输出：[CLS]篇章[SEP]问题[SEP]答案[SEP]
        """
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_masked_token_ids, batch_segment_ids, batch_token_ids = [], [], []
        for i in idxs:
            D = self.data[i]
            question = D['question']
            answers = [p['answer'] for p in D['passages'] if p['answer']]
            passage = np.random.choice(D['passages'])['passage']
            passage = re.sub(u' |、|；|，', ',', passage)
            final_answer = ''
            for answer in answers:
                if all([a in passage[:max_p_len - 2] for a in answer.split(' ')]):
                    final_answer = answer.replace(' ', ',')
                    break
            p_token_ids, _ = tokenizer.encode(passage, max_length=max_p_len)
            q_token_ids, _ = tokenizer.encode(question, max_length=max_q_len + 1)
            a_token_ids, _ = tokenizer.encode(final_answer, max_length=max_a_len + 1)
            pq_token_ids = p_token_ids + q_token_ids[1:]
            masked_token_ids = pq_token_ids + [tokenizer._token_mask_id] * max_a_len
            segment_ids = [0] * len(pq_token_ids) + [1] * max_a_len
            token_ids = pq_token_ids + a_token_ids[1:]
            token_ids += [0] * (max_a_len + 1 - len(a_token_ids))
            batch_masked_token_ids.append(masked_token_ids)
            batch_segment_ids.append(segment_ids)
            batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_masked_token_ids = sequence_padding(batch_masked_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_token_ids = sequence_padding(batch_token_ids)
                yield [batch_masked_token_ids, batch_segment_ids], batch_token_ids
                batch_masked_token_ids, batch_segment_ids, batch_token_ids = [], [], []


model = build_bert_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
)
model.summary()


def masked_cross_entropy(y_true, y_pred):
    """叉熵作为loss，并mask掉输入部分的预测
    """
    segment_ids = search_layer(y_pred, 'Input-Segment').output
    y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
    y_mask_1 = K.cast(segment_ids, K.floatx())
    y_mask_2 = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_mask = y_mask_1 * y_mask_2
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
    return cross_entropy


model.compile(loss=masked_cross_entropy, optimizer=Adam(1e-5))


def get_ngram_set(x, n):
    """生成ngram合集，返回结果格式是:
    {(n-1)-gram: set([n-gram的第n个字集合])}
    """
    result = {}
    for i in range(len(x) - n + 1):
        k = tuple(x[i:i + n])
        if k[:-1] not in result:
            result[k[:-1]] = set()
        result[k[:-1]].add(k[-1])
    return result


def gen_answer(question, passages):
    """由于是MLM模型，所以可以直接argmax解码。
    """
    all_p_token_ids, token_ids, segment_ids = [], [], []
    for passage in passages:
        passage = re.sub(u' |、|；|，', ',', passage)
        p_token_ids = tokenizer.encode(passage, max_length=max_p_len)[0]
        q_token_ids = tokenizer.encode(question, max_length=max_q_len + 1)[0]
        pq_token_ids = p_token_ids + q_token_ids[1:]
        all_p_token_ids.append(p_token_ids[1:])
        token_ids.append(pq_token_ids + [tokenizer._token_mask_id] * max_a_len)
        segment_ids.append([0] * len(pq_token_ids) + [1] * max_a_len)
    token_ids = sequence_padding(token_ids)
    segment_ids = sequence_padding(segment_ids)
    probas = model.predict([token_ids, segment_ids])
    results = {}
    for t, s, p in zip(all_p_token_ids, segment_ids, probas):
        p = p[s == 1]
        a, score = tuple(), 0.
        for i in range(max_a_len):
            idxs = list(get_ngram_set(t, i + 1)[a])
            if tokenizer._token_sep_id not in idxs:
                idxs.append(tokenizer._token_sep_id)
            # pi是将passage以外的token的概率置零
            pi = np.zeros_like(p[i])
            pi[idxs] = p[i, idxs]
            a = a + (pi.argmax(), )
            score += pi.max()
            if a[-1] == tokenizer._token_sep_id:
                break
        score = score / (i + 1)
        a = tokenizer.decode(a)
        if a:
            results[a] = results.get(a, []) + [score]
    results = {
        k: (np.array(v)**2).sum() / (sum(v) + 1)
        for k, v in results.items()
    }
    return results


def max_in_dict(d):
    if d:
        return sorted(d.items(), key=lambda s: -s[1])[0][0]


def predict_to_file(data, filename):
    """将预测结果输出到文件，方便评估
    """
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q_text = d['question']
            p_texts = [p['passage'] for p in d['passages']]
            a = gen_answer(q_text, p_texts)
            a = max_in_dict(a)
            if a:
                s = u'%s\t%s\n' % (d['id'], a)
            else:
                s = u'%s\t\n' % (d['id'])
            f.write(s)
            f.flush()


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')


if __name__ == '__main__':

    evaluator = Evaluate()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator])

else:

    model.load_weights('./best_model.weights')
