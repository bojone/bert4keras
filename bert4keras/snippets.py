#! -*- coding: utf-8 -*-
# 代码合集

import six
import logging
import numpy as np
import re


is_py2 = six.PY2

if not is_py2:
    basestring = str


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, basestring)


def strQ2B(ustring):
    """全角符号转对应的半角符号
    """
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        # 全角空格直接转换
        if inside_code == 12288:
            inside_code = 32
        # 全角字符（除空格）根据关系转化
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring


def string_matching(s, keywords):
    """判断s是否至少包含keywords中的至少一个字符串
    """
    for k in keywords:
        if re.search(k, s):
            return True
    return False


class Progress:
    """显示进度，自己简单封装，比tqdm更可控一些
    iterable: 可迭代的对象；
    period: 显示进度的周期；
    steps: iterable可迭代的总步数，相当于len(iterable)
    """
    def __init__(self, iterable, period=1, steps=None, desc=None):
        self.iterable = iterable
        self.period = period
        if hasattr(iterable, '__len__'):
            self.steps = len(iterable)
        else:
            self.steps = steps
        self.desc = desc
        if self.steps:
            self._format_ = u'%s/%s passed' % ('%s', self.steps)
        else:
            self._format_ = u'%s passed'
        if self.desc:
            self._format_ = self.desc + ' - ' + self._format_
        self.logger = logging.getLogger()

    def __iter__(self):
        for i, j in enumerate(self.iterable):
            if (i + 1) % self.period == 0:
                self.logger.info(self._format_ % (i + 1))
            yield j


def parallel_apply(func,
                   iterable,
                   workers,
                   max_queue_size,
                   callback=None,
                   dummy=False):
    """多进程或多线程地将func应用到iterable的每个元素中。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是
    输出可能是func(c), func(a), func(b)。
    参数：
        dummy: False是多进程/线性，True则是多线程/线性；
        callback: 处理单个输出的回调函数；
    """
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue

    in_queue, out_queue = Queue(max_queue_size), Queue()

    def worker_step(in_queue, out_queue):
        # 单步函数包装成循环执行
        while True:
            d = in_queue.get()
            r = func(d)
            out_queue.put(r)

    # 启动多进程/线程
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    if callback is None:
        results = []

    # 后处理函数
    def process_out_queue():
        out_count = 0
        for _ in range(out_queue.qsize()):
            d = out_queue.get()
            out_count += 1
            if callback is None:
                results.append(d)
            else:
                callback(d)
        return out_count

    # 存入数据，取出结果
    in_count, out_count = 0, 0
    for d in iterable:
        in_count += 1
        while True:
            try:
                in_queue.put(d, block=False)
                break
            except six.moves.queue.Full:
                out_count += process_out_queue()
        if in_count % max_queue_size == 0:
            out_count += process_out_queue()

    while out_count != in_count:
        out_count += process_out_queue()

    pool.terminate()

    if callback is None:
        return results


def get_all_attributes(something):
    """获取类下的所有属性和方法
    """
    return {
        name: getattr(something, name)
        for name in dir(something) if name[0] != '_'
    }


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] * (length - len(x))])
        if len(x) < length else x[:length] for x in inputs
    ])
    return outputs


def is_one_of(x, ys):
    """判断x是否在ys之中
    等价于x in ys，但有些情况下x in ys会报错
    """
    for y in ys:
        if x is y:
            return True
    return False


class DataGenerator(object):
    """数据生成器模版
    """
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d
