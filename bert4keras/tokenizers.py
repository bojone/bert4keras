#! -*- coding: utf-8 -*-
# 工具函数

import unicodedata, re
from bert4keras.snippets import is_string, is_py2
from bert4keras.snippets import open


def load_vocab(dict_path, encoding='utf-8', simplified=False, startswith=None):
    """从bert的词典文件中读取词典
    """
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    if simplified:  # 过滤冗余部分token
        new_token_dict, keep_tokens = {}, []
        startswith = startswith or []
        for t in startswith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict:
                keep = True
                if len(t) > 1:
                    for c in Tokenizer.stem(t):
                        if (
                            Tokenizer._is_cjk_character(c) or
                            Tokenizer._is_punctuation(c)
                        ):
                            keep = False
                            break
                if keep:
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict


def save_vocab(dict_path, token_dict, encoding='utf-8'):
    """将词典（比如精简过的）保存为文件
    """
    with open(dict_path, 'w', encoding=encoding) as writer:
        for k, v in sorted(token_dict.items(), key=lambda s: s[1]):
            writer.write(k + '\n')


class BasicTokenizer(object):
    """分词器基类
    """
    def __init__(self, token_start='[CLS]', token_end='[SEP]'):
        """初始化
        """
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._token_start = token_start
        self._token_end = token_end

    def tokenize(self, text, max_length=None):
        """分词函数
        """
        tokens = self._tokenize(text)
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)

        if max_length is not None:
            index = int(self._token_end is not None) + 1
            self.truncate_sequence(max_length, tokens, None, -index)

        return tokens

    def token_to_id(self, token):
        """token转换为对应的id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        """token序列转换为对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def truncate_sequence(
        self, max_length, first_sequence, second_sequence=None, pop_index=-1
    ):
        """截断总长度
        """
        if second_sequence is None:
            second_sequence = []

        while True:
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= max_length:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(pop_index)
            else:
                second_sequence.pop(pop_index)

    def encode(
        self,
        first_text,
        second_text=None,
        max_length=None,
        first_length=None,
        second_length=None
    ):
        """输出文本对应token id和segment id
        如果传入first_length，则强行padding第一个句子到指定长度；
        同理，如果传入second_length，则强行padding第二个句子到指定长度。
        """
        if is_string(first_text):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            idx = int(bool(self._token_start))
            second_tokens = self.tokenize(second_text)[idx:]
        else:
            second_tokens = second_text

        if max_length is not None:
            self.truncate_sequence(max_length, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)
        if first_length is not None:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self._token_pad_id] *
                                   (first_length - len(first_token_ids)))
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            second_token_ids = self.tokens_to_ids(second_tokens)
            if second_length is not None:
                second_token_ids = second_token_ids[:second_length]
                second_token_ids.extend([self._token_pad_id] *
                                        (second_length - len(second_token_ids)))
            second_segment_ids = [1] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def id_to_token(self, i):
        """id序列为对应的token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id序列转换为对应的token序列
        """
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids):
        """转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """基本分词函数
        """
        raise NotImplementedError


class Tokenizer(BasicTokenizer):
    """Bert原生分词器
    纯Python实现，代码修改自keras_bert的tokenizer实现
    """
    def __init__(self, token_dict, do_lower_case=False, *args, **kwargs):
        """初始化
        """
        super(Tokenizer, self).__init__(*args, **kwargs)
        if is_string(token_dict):
            token_dict = load_vocab(token_dict) # 返回一个字典d, d[token] = id

        self._do_lower_case = do_lower_case
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()} # inverse的字典,原来是key: value, 变成value: key
        self._vocab_size = len(token_dict)

        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                # getattr(object, name[, default]) -> value, get attribute
                # getattr(x, 'y') is equivalent to x.y
                _token_id = token_dict[getattr(self, '_token_%s' % token)]
                # setattr(x, 'y', v) is equivalent to 'x.y = v', set attribute
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass

    def token_to_id(self, token):
        """token转换为对应的id
        使用dict.get(key)而不用dict[key]的原因是get方法在key不存在时返回预设的默认值
        https://stackoverflow.com/questions/11041405/why-dict-getkey-instead-of-dictkey
        """
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i):
        """id转换为对应的token
        """
        return self._token_dict_inv[i]

    def decode(self, ids, tokens=None):
        """转为可读文本
        """
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token

        text = re.sub(' +', ' ', text)
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)

        return text.strip()

    def _tokenize(self, text):
        """基本分词函数
        """
        if self._do_lower_case:
            if is_py2:
                text = unicode(text)
            text = unicodedata.normalize('NFD', text)
            text = ''.join([
                ch for ch in text if unicodedata.category(ch) != 'Mn'
            ])
            text = text.lower()

        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))

        return tokens

    def _word_piece_tokenize(self, word):
        """word内分成subword
        """
        if word in self._token_dict:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'。
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\xb7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号

        这句话的意思是, 字符串不为空并且开头结尾是[]
        其中, bool()的含义是判断字符串是否为空
        Example:
        >>> x = 'GeeksforGeeks'
        >>> print(bool(x))
        True
        https://www.geeksforgeeks.org/bool-in-python/
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if is_py2:
            text = unicode(text)

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
                ch = ch.lower()
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


class SpTokenizer(BasicTokenizer):
    """基于SentencePiece模型的封装，使用上跟Tokenizer基本一致。
    """
    def __init__(self, sp_model_path, *args, **kwargs):
        super(SpTokenizer, self).__init__(*args, **kwargs)
        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sp_model_path)
        self._token_pad = self.sp_model.id_to_piece(self.sp_model.pad_id())
        self._token_unk = self.sp_model.id_to_piece(self.sp_model.unk_id())
        self._vocab_size = self.sp_model.get_piece_size()

        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                _token = getattr(self, '_token_%s' % token)
                _token_id = self.sp_model.piece_to_id(_token)
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass

    def token_to_id(self, token):
        """token转换为对应的id
        """
        return self.sp_model.piece_to_id(token)

    def id_to_token(self, i):
        """id转换为对应的token
        """
        if i < self._vocab_size:
            return self.sp_model.id_to_piece(i)
        else:
            return ''

    def decode(self, ids):
        """转为可读文本
        """
        ids = [i for i in ids if self._is_decodable(i)]
        text = self.sp_model.decode_ids(ids)
        return text.decode('utf-8') if is_py2 else text

    def _tokenize(self, text):
        """基本分词函数
        """
        tokens = self.sp_model.encode_as_pieces(text)
        return tokens

    def _is_special(self, i):
        """判断是不是有特殊含义的符号
        """
        return self.sp_model.is_control(i) or \
            self.sp_model.is_unknown(i) or \
            self.sp_model.is_unused(i)

    def _is_decodable(self, i):
        """判断是否应该被解码输出
        """
        return (i < self._vocab_size) and not self._is_special(i)
