#! -*- coding: utf-8 -*-
# 用seq2seq的方式做阅读理解任务
# 数据集和评测同 https://github.com/bojone/dgcnn_for_reading_comprehension
# 8个epoch后在valid上能达到约0.77的分数
# (Accuracy=0.7259005836184343  F1=0.813860036706151    Final=0.7698803101622926)

import json, os, re
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
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
    random_order = list(range(len(sogou_data)))
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
        """单条样本格式：[CLS]篇章[SEP]问题[SEP]答案[SEP]
        """
        batch_token_ids, batch_segment_ids = [], []
        for is_end, D in self.sample(random):
            question = D['question']
            answers = [p['answer'] for p in D['passages'] if p['answer']]
            passage = np.random.choice(D['passages'])['passage']
            passage = re.sub(u' |、|；|，', ',', passage)
            final_answer = ''
            for answer in answers:
                if all([
                    a in passage[:max_p_len - 2] for a in answer.split(' ')
                ]):
                    final_answer = answer.replace(' ', ',')
                    break
            qa_token_ids, qa_segment_ids = tokenizer.encode(
                question, final_answer, max_length=max_qa_len + 1
            )
            p_token_ids, p_segment_ids = tokenizer.encode(
                passage, max_length=max_p_len
            )
            token_ids = p_token_ids + qa_token_ids[1:]
            segment_ids = p_segment_ids + qa_segment_ids[1:]
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


class ReadingComprehension(AutoRegressiveDecoder):
    """beam search解码来生成答案
    passages为多篇章组成的list，从多篇文章中自动决策出最优的答案，
    如果没答案，则返回空字符串。
    mode是extractive时，按照抽取式执行，即答案必须是原篇章的一个片段。
    """
    def __init__(self, mode='extractive', **kwargs):
        super(ReadingComprehension, self).__init__(**kwargs)
        self.mode = mode

    def get_ngram_set(self, x, n):
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

    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        topk = len(inputs[0])
        all_token_ids, all_segment_ids = [], []
        for token_ids in inputs:  # inputs里每个元素都代表一个篇章
            token_ids = np.concatenate([token_ids, output_ids], 1)
            segment_ids = np.zeros_like(token_ids)
            if step > 0:
                segment_ids[:, -output_ids.shape[1]:] = 1
            all_token_ids.extend(token_ids)
            all_segment_ids.extend(segment_ids)
        padded_all_token_ids = sequence_padding(all_token_ids)
        padded_all_segment_ids = sequence_padding(all_segment_ids)
        probas = model.predict([padded_all_token_ids, padded_all_segment_ids])
        probas = [
            probas[i, len(ids) - 1] for i, ids in enumerate(all_token_ids)
        ]
        probas = np.array(probas).reshape((len(inputs), topk, -1))
        if step == 0:
            # 这一步主要是排除没有答案的篇章
            # 如果一开始最大值就为end_id，那说明该篇章没有答案
            argmax = probas[:, 0].argmax(axis=1)
            available_idxs = np.where(argmax != self.end_id)[0]
            if len(available_idxs) == 0:
                scores = np.zeros_like(probas[0])
                scores[:, self.end_id] = 1
                return scores
            else:
                for i in np.where(argmax == self.end_id)[0]:
                    inputs[i][:, 0] = -1  # 无答案篇章首位标记为-1
                probas = probas[available_idxs]
                inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        if self.mode == 'extractive':
            # 如果是抽取式，那么答案必须是篇章的一个片段
            # 那么将非篇章片段的概率值全部置0
            new_probas = np.zeros_like(probas)
            ngrams = {}
            for token_ids in inputs:
                token_ids = token_ids[0]
                sep_idx = np.where(token_ids == tokenizer._token_end_id)[0][0]
                p_token_ids = token_ids[1:sep_idx]
                for k, v in self.get_ngram_set(p_token_ids, step + 1).items():
                    ngrams[k] = ngrams.get(k, set()) | v
            for i, ids in enumerate(output_ids):
                available_idxs = ngrams.get(tuple(ids), set())
                available_idxs.add(tokenizer._token_end_id)
                available_idxs = list(available_idxs)
                new_probas[:, i, available_idxs] = probas[:, i, available_idxs]
            probas = new_probas
        return (probas**2).sum(0) / (probas.sum(0) + 1)  # 某种平均投票方式

    def answer(self, question, passages, topk=1):
        token_ids = []
        for passage in passages:
            passage = re.sub(u' |、|；|，', ',', passage)
            p_token_ids = tokenizer.encode(passage, max_length=max_p_len)[0]
            q_token_ids = tokenizer.encode(question,
                                           max_length=max_q_len + 1)[0]
            token_ids.append(p_token_ids + q_token_ids[1:])
        output_ids = self.beam_search(token_ids, topk)  # 基于beam search
        return tokenizer.decode(output_ids)


reader = ReadingComprehension(
    start_id=None,
    end_id=tokenizer._token_end_id,
    maxlen=max_a_len,
    mode='extractive'
)


def predict_to_file(data, filename, topk=1):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q_text = d['question']
            p_texts = [p['passage'] for p in d['passages']]
            a = reader.answer(q_text, p_texts, topk)
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

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')
