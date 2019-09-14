#! -*- coding: utf-8 -*-
# 工具函数

import unicodedata
import codecs


class SimpleTokenizer:
    """简单的分词器，直接将文本分割为单字符序列，
    专为中文处理设计，原则上只适用于中文模型。
    """
    def __init__(self, token_dict):
        """初始化词典
        """
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}

    def _is_space(self, c):
        """判断是否为空格
        """
        return c == ' ' or c == '\n' or c == '\r' or c == '\t' or \
            unicodedata.category(c) == 'Zs'
    
    def _is_special(self, c):
        """判断是否带方括号的特殊标记
        """
        return bool(c) and (c[0] == '[') and (c[-1] == ']')

    def tokenize(self, text, add_cls=True, add_sep=True):
        """按字分割
        """
        R = []
        if add_cls:
            R.append('[CLS]')
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        if add_sep:
            R.append('[SEP]')
        return R

    def encode(self, first, second=None, first_length=None):
        """输出文本对应token id和segment id
        如果传入first_length，则强行padding第一个句子到指定长度
        """
        token_ids, segment_ids = [], []
        token_ids.extend([self._token_dict[c] for c in self.tokenize(first)])
        segment_ids.extend([0] * (len(first) + 2))
        if first_length is not None and len(token_ids) < first_length + 2:
            token_ids.extend([0] * (first_length + 2 - len(token_ids)))
            segment_ids.extend([0] * (first_length + 2 - len(segment_ids)))
        if second is not None:
            token_ids.extend([
                self._token_dict[c]
                for c in self.tokenize(second, add_cls=False)
            ])
            segment_ids.extend([1] * (len(second) + 1))
        return token_ids, segment_ids
    
    def decode(self, token_ids, join_str=''):
        """简单的词id序列转文本函数
        """
        tokens = []
        for i in token_ids:
            t = self._token_dict_inv.get(i, '')
            if t == '[unused1]':
                tokens.append(' ')
            elif not self._is_special(t):
                tokens.append(t)
        return join_str.join(tokens)
        

def load_vocab(dict_path):
    """从bert的词典文件中读取词典
    """
    token_dict = {}
    with codecs.open(dict_path, encoding='utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict
