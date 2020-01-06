#! -*- coding: utf-8 -*-
# 用seq2seq的方式阅读理解任务
# 数据集和评测同 https://github.com/bojone/dgcnn_for_reading_comprehension
# 8个epoch后在valid上能达到约0.77的分数
# (Accuracy=0.7259005836184343	F1=0.813860036706151	Final=0.7698803101622926)

import json, os
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
import codecs, re
from tqdm import tqdm


max_p_len = 256
max_q_len = 64
max_a_len = 32
max_qa_len = max_q_len + max_a_len
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
        """单条样本格式：[CLS]篇章[SEP]问题[SEP]答案[SEP]
        """
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids = [], []
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
            qa_token_ids, qa_segment_ids = tokenizer.encode(
                question, final_answer, max_length=max_qa_len + 1)
            p_token_ids, p_segment_ids = tokenizer.encode(passage,
                                                          max_length=max_p_len)
            token_ids = p_token_ids + qa_token_ids[1:]
            segment_ids = p_segment_ids + qa_segment_ids[1:]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


model = build_bert_model(
    config_path,
    checkpoint_path,
    application='seq2seq',
    keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
)
model.summary()

# 交叉熵作为loss，并mask掉输入部分的预测
y_in = model.input[0][:, 1:]  # 目标tokens
y_mask = model.input[1][:, 1:]
y = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))


def get_ngram_set(x, n):
    """生成ngram合集，返回结果格式是:
    {(n-1)-gram: set([n-gram的第n个字集合])}
    """
    result = {}
    for i in range(len(x) - n + 1):
        k = tuple(x[i: i + n])
        if k[:-1] not in result:
            result[k[:-1]] = set()
        result[k[:-1]].add(k[-1])
    return result


def gen_answer(question, passages, topk=2, mode='extractive'):
    """beam search解码来生成答案
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索。
    passages为多篇章组成的list，从多篇文章中自动决策出最优的答案，
    如果没答案，则返回空字符串。
    mode是extractive时，按照抽取式执行，即答案必须是原篇章的一个片段。
    """
    token_ids, segment_ids = [], []
    for passage in passages:
        passage = re.sub(u' |、|；|，', ',', passage)
        p_token_ids = tokenizer.encode(passage, max_length=max_p_len)[0]
        q_token_ids = tokenizer.encode(question, max_length=max_q_len + 1)[0]
        token_ids.append(p_token_ids + q_token_ids[1:])
        segment_ids.append([0] * len(token_ids[-1]))
    target_ids = [[] for _ in range(topk)]  # 候选答案id
    target_scores = [0] * topk  # 候选答案分数
    for i in range(max_a_len):  # 强制要求输出不超过max_a_len字
        _target_ids, _segment_ids = [], []
        # 篇章与候选答案组合
        for tids, sids in zip(token_ids, segment_ids):
            for t in target_ids:
                _target_ids.append(tids + t)
                _segment_ids.append(sids + [1] * len(t))
        _padded_target_ids = sequence_padding(_target_ids)
        _padded_segment_ids = sequence_padding(_segment_ids)
        _probas = model.predict([_padded_target_ids, _padded_segment_ids
                                 ])[..., 3:]  # 直接忽略[PAD], [UNK], [CLS]
        _probas = [
            _probas[j, len(ids) - 1] for j, ids in enumerate(_target_ids)
        ]
        _probas = np.array(_probas).reshape((len(token_ids), topk, -1))
        if i == 0:
            # 这一步主要是排除没有答案的篇章
            # 如果开始[SEP]为最大值，那说明该篇章没有答案
            _probas_argmax = _probas[:, 0].argmax(axis=1)
            _available_idxs = np.where(_probas_argmax != 0)[0]
            if len(_available_idxs) == 0:
                return ''
            else:
                _probas = _probas[_available_idxs]
                token_ids = [token_ids[j] for j in _available_idxs]
                segment_ids = [segment_ids[j] for j in _available_idxs]
        if mode == 'extractive':
            # 如果是抽取式，那么答案必须是篇章的一个片段
            # 那么将非篇章片段的概率值全部置0
            _zeros = np.zeros_like(_probas)
            _ngrams = {}
            for p_token_ids in token_ids:
                for k, v in get_ngram_set(p_token_ids, i + 1).items():
                    _ngrams[k] = _ngrams.get(k, set()) | v
            for j, t in enumerate(target_ids):
                _available_idxs = _ngrams.get(tuple(t), set())
                _available_idxs.add(token_dict['[SEP]'])
                _available_idxs = [k - 3 for k in _available_idxs]
                _zeros[:, j, _available_idxs] = _probas[:, j, _available_idxs]
            _probas = _zeros
        _probas = (_probas**2).sum(0) / (_probas.sum(0) + 1)  # 某种平均投票方式
        _log_probas = np.log(_probas + 1e-6)  # 取对数，方便计算
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            # 预测第一个字的时候，输入的topk事实上都是同一个，
            # 所以只需要看第一个，不需要遍历后面的。
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:]  # 从中选出新的topk
        target_ids = [_candidate_ids[k] for k in _topk_arg]
        target_scores = [_candidate_scores[k] for k in _topk_arg]
        best_one = np.argmax(target_scores)
        if target_ids[best_one][-1] == 3:
            return tokenizer.decode(target_ids[best_one])
    # 如果max_a_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def predict_to_file(data, filename, topk=2, mode='extractive'):
    """将预测结果输出到文件，方便评估
    """
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q_text = d['question']
            p_texts = [p['passage'] for p in d['passages']]
            a = gen_answer(q_text, p_texts, topk, mode)
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
