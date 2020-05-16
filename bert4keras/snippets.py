#! -*- coding: utf-8 -*-
# 代码合集

import six
import logging
import numpy as np
import re
import sys

_open_ = open
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


def convert_to_unicode(text, encoding='utf-8', errors='ignore'):
    """字符串转换为unicode格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, str):
            text = text.decode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


def convert_to_str(text, encoding='utf-8', errors='ignore'):
    """字符串转换为str格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, unicode):
            text = text.encode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


class open:
    """模仿python自带的open函数，主要是为了同时兼容py2和py3
    """
    def __init__(self, name, mode='r', encoding=None, errors='ignore'):
        if is_py2:
            self.file = _open_(name, mode)
        else:
            self.file = _open_(name, mode, encoding=encoding, errors=errors)
        self.encoding = encoding
        self.errors = errors

    def __iter__(self):
        for l in self.file:
            if self.encoding:
                l = convert_to_unicode(l, self.encoding, self.errors)
            yield l

    def read(self):
        text = self.file.read()
        if self.encoding:
            text = convert_to_unicode(text, self.encoding, self.errors)
        return text

    def write(self, text):
        if self.encoding:
            text = convert_to_str(text, self.encoding, self.errors)
        self.file.write(text)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


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


def parallel_apply(
    func, iterable, workers, max_queue_size, callback=None, dummy=False
):
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


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


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
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d


class ViterbiDecoder(object):
    """Viterbi解码算法基类
    """
    def __init__(self, trans, starts=None, ends=None):
        self.trans = trans
        self.num_labels = len(trans)
        self.non_starts = []
        self.non_ends = []
        if starts is not None:
            for i in range(self.num_labels):
                if i not in starts:
                    self.non_starts.append(i)
        if ends is not None:
            for i in range(self.num_labels):
                if i not in ends:
                    self.non_ends.append(i)

    def decode(self, nodes):
        """nodes.shape=[seq_len, num_labels]
        """
        # 预处理
        nodes[0, self.non_starts] -= np.inf
        nodes[-1, self.non_ends] -= np.inf

        # 动态规划
        labels = np.arange(self.num_labels).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        paths = labels
        for l in range(1, len(nodes)):
            M = scores + self.trans + nodes[l].reshape((1, -1))
            idxs = M.argmax(0)
            scores = M.max(0).reshape((-1, 1))
            paths = np.concatenate([paths[:, idxs], labels], 0)

        # 最优路径
        return paths[:, scores[:, 0].argmax()]


def softmax(x, axis=-1):
    """numpy版softmax
    """
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)


class AutoRegressiveDecoder(object):
    """通用自回归生成模型解码基类
    包含beam search和random sample两种策略
    """
    def __init__(self, start_id, end_id, maxlen, minlen=None):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen or 1
        if start_id is None:
            self.first_output_ids = np.empty((1, 0), dtype=int)
        else:
            self.first_output_ids = np.array([[self.start_id]])

    @staticmethod
    def set_rtype(default='probas'):
        """用来给predict方法加上rtype参数，并作相应的处理
        """
        def actual_decorator(predict):
            def new_predict(self, inputs, output_ids, step, rtype=default):
                assert rtype in ['probas', 'logits']
                result = predict(self, inputs, output_ids, step)
                if default == 'probas':
                    if rtype == 'probas':
                        return result
                    else:
                        return np.log(result + 1e-12)
                else:
                    if rtype == 'probas':
                        return softmax(result, -1)
                    else:
                        return result

            return new_predict

        return actual_decorator

    def predict(self, inputs, output_ids, step, rtype='logits'):
        """用户需自定义递归预测函数
        rtype为字符串logits或probas，用户定义的时候，应当根据rtype来
        返回不同的结果，rtype=probas时返回归一化的概率，rtype=logits时
        则返回softmax前的结果或者概率对数。
        """
        raise NotImplementedError

    def beam_search(self, inputs, topk):
        """beam search解码
        说明：这里的topk即beam size；
        返回：最优解码序列。
        """
        inputs = [np.array([i]) for i in inputs]
        output_ids, output_scores = self.first_output_ids, np.zeros(1)
        for step in range(self.maxlen):
            scores = self.predict(inputs, output_ids, step, 'logits')  # 计算当前得分
            if step == 0:  # 第1步预测后将输入重复topk次
                inputs = [np.repeat(i, topk, axis=0) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
            indices = scores.argpartition(-topk, axis=None)[-topk:]  # 仅保留topk
            indices_1 = indices // scores.shape[1]  # 行索引
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
            output_ids = np.concatenate([output_ids[indices_1], indices_2],
                                        1)  # 更新输出
            output_scores = np.take_along_axis(
                scores, indices, axis=None
            )  # 更新得分
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best_one = output_scores.argmax()  # 得分最大的那个
                if indices_2[best_one, 0] == self.end_id:  # 如果已经终止
                    return output_ids[best_one]  # 直接输出
                else:  # 否则，只保留未完成部分
                    flag = (indices_2[:, 0] != self.end_id)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        topk = flag.sum()  # topk相应变化
        # 达到长度直接输出
        return output_ids[output_scores.argmax()]

    def random_sample(self, inputs, n, topk=None, topp=None):
        """随机采样n个结果
        说明：非None的topk表示每一步只从概率最高的topk个中采样；而非None的topp
             表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
        返回：n个解码序列组成的list。
        """
        inputs = [np.array([i]) for i in inputs]
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            probas = self.predict(inputs, output_ids, step, 'probas')  # 计算当前概率
            probas /= probas.sum(axis=1, keepdims=True)  # 确保归一化
            if step == 0:  # 第1步预测后将结果重复n次
                probas = np.repeat(probas, n, axis=0)
                inputs = [np.repeat(i, n, axis=0) for i in inputs]
                output_ids = np.repeat(output_ids, n, axis=0)
            if topk is not None:
                k_indices = probas.argpartition(-topk,
                                                axis=1)[:, -topk:]  # 仅保留topk
                probas = np.take_along_axis(probas, k_indices, axis=1)  # topk概率
                probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
            if topp is not None:
                p_indices = probas.argsort(axis=1)[:, ::-1]  # 从高到低排序
                probas = np.take_along_axis(probas, p_indices, axis=1)  # 排序概率
                cumsum_probas = np.cumsum(probas, axis=1)  # 累积概率
                flag = np.roll(cumsum_probas >= topp, 1, axis=1)  # 标记超过topp的部分
                flag[:, 0] = False  # 结合上面的np.roll，实现平移一位的效果
                probas[flag] = 0  # 后面的全部置零
                probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
            sample_func = lambda p: np.random.choice(len(p), p=p)  # 按概率采样函数
            sample_ids = np.apply_along_axis(sample_func, 1, probas)  # 执行采样
            sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
            if topp is not None:
                sample_ids = np.take_along_axis(
                    p_indices, sample_ids, axis=1
                )  # 对齐原id
            if topk is not None:
                sample_ids = np.take_along_axis(
                    k_indices, sample_ids, axis=1
                )  # 对齐原id
            output_ids = np.concatenate([output_ids, sample_ids], 1)  # 更新输出
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                flag = (sample_ids[:, 0] == self.end_id)  # 标记已完成序列
                if flag.any():  # 如果有已完成的
                    for ids in output_ids[flag]:  # 存好已完成序列
                        results.append(ids)
                    flag = (flag == False)  # 标记未完成序列
                    inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
                    output_ids = output_ids[flag]  # 只保留未完成部分候选集
                    if len(output_ids) == 0:
                        break
        # 如果还有未完成序列，直接放入结果
        for ids in output_ids:
            results.append(ids)
        # 返回结果
        return results


def insert_arguments(**arguments):
    """装饰器，为类方法增加参数
    （主要用于类的__init__方法）
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def delete_arguments(*arguments):
    """装饰器，为类方法删除参数
    （主要用于类的__init__方法）
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k in arguments:
                if k in kwargs:
                    raise TypeError(
                        '%s got an unexpected keyword argument \'%s\'' %
                        (self.__class__.__name__, k)
                    )
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def groupby(iterable, key=None):
    """类似itertools.groupby，但这里的key是iterable对象
    """
    if key is None:
        key = iterable

    result = []
    for i, (k, v) in enumerate(zip(key, iterable)):
        if i == 0:
            result.append((k, [v]))
            last_k = k
        else:
            if k == last_k:
                result[-1][1].append(v)
            else:
                result.append((k, [v]))
                last_k = k

    return result


class Hook:
    """注入uniout模块，实现import时才触发
    """
    def __init__(self, module):
        self.module = module

    def __getattr__(self, attr):
        """使得 from bert4keras.backend import uniout
        等效于 import uniout （自动识别Python版本，Python3
        下则无操作。）
        """
        if attr == 'uniout':
            if is_py2:
                import uniout
        else:
            return getattr(self.module, attr)


Hook.__name__ = __name__
sys.modules[__name__] = Hook(sys.modules[__name__])
del Hook
