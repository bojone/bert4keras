#! -*- coding: utf-8 -*-

import sys
import warnings


__version__ = '0.6.2'


class Legacy1:
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
        warnings.warn('bert4keras.bert has been renamed as bert4keras.models.')
        warnings.warn('please use bert4keras.models.')
        return getattr(self.models, attr)


Legacy1.__name__ = 'bert4keras.bert'
sys.modules[Legacy1.__name__] = Legacy1()
del Legacy1


class Legacy2:
    """向后兼容
    """
    def __init__(self):
        import bert4keras.tokenizers
        self.tokenizers = bert4keras.tokenizers
        self.__all__ = [k for k in dir(self.tokenizers) if k[0] != '_']

    def __getattr__(self, attr):
        """使得 from bert4keras.tokenizer import xxx
        等价于from bert4keras.tokenizers import xxx
        """
        warnings.warn('bert4keras.tokenizer has been renamed as bert4keras.tokenizers.')
        warnings.warn('please use bert4keras.tokenizers.')
        return getattr(self.tokenizers, attr)


Legacy2.__name__ = 'bert4keras.tokenizer'
sys.modules[Legacy2.__name__] = Legacy2()
del Legacy2
