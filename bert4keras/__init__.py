#! -*- coding: utf-8 -*-

import sys
import warnings


__version__ = '0.5.7'


class Legacy:
    """向后兼容
    """
    def __init__(self):
        import bert4keras.models
        self.models = bert4keras.models
        self.__all__ = [k for k in dir(self.models) if k[0] != '_']

    def __getattr__(self, attr):
        """使得 from bert4keras.bert import xxx
        等价于from bert4keras.models import xxx
        """
        warnings.warn(
            'bert4keras.bert has been renamed as bert4keras.models,' +
            ' and bert4keras.bert will be removed in future version.')
        return getattr(self.models, attr)


Legacy.__name__ = 'bert4keras.bert'
sys.modules[Legacy.__name__] = Legacy()
del Legacy
