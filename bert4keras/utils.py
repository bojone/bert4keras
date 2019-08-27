#! -*- coding: utf-8 -*-

import unicodedata


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

    def encode(self, first, second=None):
        """输出文本对应token id和segment id
        """
        token_ids, segment_ids = [], []
        token_ids.extend([self._token_dict[c] for c in self.tokenize(first)])
        segment_ids.extend([0] * (len(first) + 2))
        if second is not None:
            token_ids.extend([
                self._token_dict[c]
                for c in self.tokenize(second, add_cls=False)
            ])
            segment_ids.extend([1] * (len(second) + 1))
        return token_ids, segment_ids
