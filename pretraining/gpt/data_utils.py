#! -*- coding: utf-8 -*-
# 预训练语料构建

import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import numpy as np
import tensorflow as tf
from bert4keras.snippets import parallel_apply
from bert4keras.backend import K


class TrainingDataset:
    """MLM预训练数据集生成器（gpt模式，即单项语言模型）
    """
    def __init__(self, tokenizer, sequence_length=512):
        """参数说明：
            tokenizer必须是bert4keras自带的tokenizer类。
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_cls_id
        self.token_sep_id = tokenizer._token_sep_id
        self.vocab_size = tokenizer._vocab_size

    def sentence_process(self, text):
        """单个文本的处理函数
        流程：分词，然后转id。
        """
        tokens = self.tokenizer.tokenize(text=text,
                                         add_cls=False,
                                         add_sep=False)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        return token_ids

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
        token_ids = [self.token_cls_id]

        for text in texts:
            # 处理单个句子
            _token_ids = self.sentence_process(text)
            _token_ids = _token_ids[:self.sequence_length - 2]

            # 如果长度即将溢出
            if len(token_ids) + len(_token_ids) > self.sequence_length - 1:
                # 插入终止符
                token_ids.append(self.token_sep_id)
                # padding到指定长度
                token_ids = self.padding(token_ids)
                # 存储结果，并开始构建新的样本
                results.append(token_ids)
                token_ids = [self.token_cls_id]

            token_ids.extend(_token_ids)

        # 插入终止符
        token_ids.append(self.token_sep_id)
        # padding到指定长度
        token_ids = self.padding(token_ids)
        # 存储结果
        results.append(token_ids)

        return results

    def tfrecord_serialize(self, results):
        """转为tfrecord的字符串，等待写入到文件
        """
        def create_feature(x):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

        new_results = []
        for token_ids in results:
            features = {
                'token_ids': create_feature(token_ids),
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
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            segment_ids = K.zeros_like(token_ids, dtype='int64')
            x = {
                'Input-Token': token_ids,
                'Input-Segment': segment_ids,
            }
            y = {
                'lm_loss': K.zeros([1]),
                'lm_acc': K.zeros([1]),
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
    from tqdm import tqdm

    dict_path = '/home/spaces_ac_cn/chinese_L-12_H-768_A-12/vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def some_texts():
        filenames = glob.glob('/home/spaces_ac_cn/corpus/*/*/*')
        np.random.shuffle(filenames)
        count, texts = 0, []
        for filename in filenames:
            with open(filename) as f:
                for l in f:
                    l = json.loads(l)['text'].strip()
                    texts.extend(re.findall(u'.*?[\n。]+', l))
                    count += 1
                    if count == 10:  # 10篇文章合在一起再处理
                        yield texts
                        count, texts = 0, []
        if texts:
            yield texts

    TD = TrainingDataset(tokenizer, sequence_length=512)

    TD.process(
        corpus=tqdm(some_texts()),
        record_name='../corpus_tfrecord/corpus.tfrecord',
        workers=40,
        max_queue_size=4000,
    )
