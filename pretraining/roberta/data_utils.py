#! -*- coding: utf-8 -*-
# 预训练语料构建

import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import numpy as np
import tensorflow as tf
from bert4keras.snippets import parallel_apply
from bert4keras.backend import K


class TrainingDataset:
    """MLM预训练数据集生成器（roberta模式）
    """
    def __init__(self,
                 tokenizer,
                 word_segment,
                 mask_rate=0.15,
                 sequence_length=512):
        """参数说明：
            tokenizer必须是bert4keras自带的tokenizer类；
            word_segment是任意分词函数。
        """
        self.tokenizer = tokenizer
        self.word_segment = word_segment
        self.mask_rate = mask_rate
        self.sequence_length = sequence_length
        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_cls_id
        self.token_sep_id = tokenizer._token_sep_id
        self.token_mask_id = tokenizer._token_mask_id
        self.vocab_size = tokenizer._vocab_size

    def token_process(self, token_id):
        """以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def sentence_process(self, text):
        """单个文本的处理函数
        流程：分词，然后转id，按照mask_rate构建全词mask的序列
              来指定哪些token是否要被mask
        """
        words = self.word_segment(text)
        rands = np.random.random(len(words))

        token_ids, mask_ids = [], []
        for rand, word in zip(rands, words):
            word_tokens = self.tokenizer.tokenize(text=word,
                                                  add_cls=False,
                                                  add_sep=False)
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)

            if rand < self.mask_rate:
                word_mask_ids = [
                    self.token_process(i) + 1 for i in word_token_ids
                ]
            else:
                word_mask_ids = [0] * len(word_tokens)

            mask_ids.extend(word_mask_ids)

        return token_ids, mask_ids

    def padding(self, sequence, padding_value=None):
        """对单个序列进行补0
        """
        if padding_value is None:
            padding_value = self.token_pad_id

        sequence = sequence[:self.sequence_length]
        padding_length = self.sequence_length - len(sequence)
        return sequence + [padding_value] * padding_length

    def paragraph_process(self, texts):
        """texts是单句组成的list
        做法：不断塞句子，直到长度最接近sequence_length，然后补0。
        """
        results = []
        token_ids, mask_ids = [self.token_cls_id], [0]

        for text in texts:
            # 处理单个句子
            _token_ids, _mask_ids = self.sentence_process(text)
            _token_ids = _token_ids[:self.sequence_length - 2]
            _mask_ids = _mask_ids[:self.sequence_length - 2]

            # 如果长度即将溢出
            if len(mask_ids) + len(_mask_ids) > self.sequence_length - 1:
                # 插入终止符
                token_ids.append(self.token_sep_id)
                mask_ids.append(0)
                # padding到指定长度
                token_ids = self.padding(token_ids)
                mask_ids = self.padding(mask_ids, 0)
                # 存储结果，并开始构建新的样本
                results.append((token_ids, mask_ids))
                token_ids, mask_ids = [self.token_cls_id], [0]

            token_ids.extend(_token_ids)
            mask_ids.extend(_mask_ids)

        return results

    def tfrecord_serialize(self, results):
        """转为tfrecord的字符串，等待写入到文件
        """
        def create_feature(x):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

        new_results = []
        for token_ids, mask_ids in results:
            features = {
                'token_ids': create_feature(token_ids),
                'mask_ids': create_feature(mask_ids),
            }
            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            tf_serialized = tf_example.SerializeToString()
            new_results.append(tf_serialized)

        return new_results

    def process(self, corpus, record_name, workers=8, max_queue_size=2000):
        """处理输入语料（corpus），最终转为tfrecord格式（record_name）
        自带多进程支持，如果cpu核心数多，请加大workers和max_queue_size。
        """
        writer = tf.io.TFRecordWriter(record_name)
        globals()['count'] = 0

        def write_to_tfrecord(results):
            globals()['count'] += len(results)
            for tf_serialized in results:
                writer.write(tf_serialized)

        def paragraph_process(texts):
            results = self.paragraph_process(texts)
            results = self.tfrecord_serialize(results)
            return results

        parallel_apply(
            func=paragraph_process,
            iterable=corpus,
            workers=workers,
            max_queue_size=max_queue_size,
            callback=write_to_tfrecord,
        )

        writer.close()
        print('write %s examples into %s' % (count, record_name))

    @staticmethod
    def load_tfrecord(record_names, sequence_length, batch_size):
        """加载处理成tfrecord格式的语料
        """
        if not isinstance(record_names, list):
            record_names = [record_names]

        dataset = tf.data.TFRecordDataset(record_names)
        FixedLenFeature = tf.io.FixedLenFeature([sequence_length], tf.int64)

        # 解析函数
        def _parse_function(serialized):
            features = {
                'token_ids': FixedLenFeature,
                'mask_ids': FixedLenFeature,
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            mask_ids = features['mask_ids']
            segment_ids = K.zeros_like(token_ids, dtype='int64')
            is_masked = K.not_equal(mask_ids, 0)
            masked_token_ids = K.switch(is_masked, mask_ids - 1, token_ids)
            x = {
                'Input-Token': masked_token_ids,
                'Input-Segment': segment_ids,
                'token_ids': token_ids,
                'is_masked': is_masked,
            }
            y = {
                'mlm_loss': K.zeros([1]),
                'mlm_acc': K.zeros([1]),
            }
            return x, y

        dataset = dataset.map(_parse_function)  # 解析
        dataset = dataset.repeat()  # 循环
        dataset = dataset.shuffle(batch_size * 1000)  # 打乱
        dataset = dataset.batch(batch_size)  # 成批

        return dataset


if __name__ == '__main__':

    # 使用测试

    from bert4keras.tokenizer import Tokenizer
    import json, glob, re
    import jieba_fast as jieba
    from tqdm import tqdm

    dict_path = '/home/spaces_ac_cn/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
    tokenizer = Tokenizer(dict_path)

    def some_texts():
        for _ in range(2): # 数据重复两遍
            filenames = glob.glob('/home/spaces_ac_cn/corpus/*/*/*')
            np.random.shuffle(filenames)
            for filename in filenames:
                with open(filename) as f:
                    for l in f:
                        l = json.loads(l)['text'].strip()
                        yield re.findall(u'.*?[\n。]+', l)

    def word_segment(text):
        return jieba.lcut(text)

    TD = TrainingDataset(tokenizer, word_segment, sequence_length=512)
    TD.process(
        corpus=tqdm(some_texts()),
        record_name='../corpus.tfrecord',
        workers=20,
        max_queue_size=20000,
    )
