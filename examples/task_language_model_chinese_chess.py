#! -*- coding: utf-8 -*-
# 用 语言模型+棋谱 的方式监督训练一个下中国象棋模型
# 介绍：https://kexue.fm/archives/7877
# 数据：https://github.com/bojone/gpt_cchess
# 模型训练可以在python2/python3进行。但是cchess模块只支持python3，
# 因此如果需要交互式体验模型棋力，那么需要在python3下进行。

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator
from keras.models import Model
from cchess import *

# 基本信息
maxlen = 512
steps_per_epoch = 1000
epochs = 10000
batch_size = 16

# bert配置
config_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """读取全局棋谱
    返回：[(棋谱, 结果)]，其中结果等于2为红方赢棋，1为和棋，
    0为黑方赢棋，-1则为无明确标注胜负。
    """
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            if not l['fen']:
                result = int(l['items'].get(u'棋局结果', -1))
                D.append((l['iccs'], result))
    return D


# 加载数据
data = load_data('/root/qipu.json')

# 建立分词器
chars = [u'[PAD]'] + list(u'0123456789abcdefghi')
token_dict = dict(zip(chars, range(len(chars))))
tokenizer = Tokenizer(token_dict)
tokenizer._token_unk_id = 0
bert_token_dict = load_vocab(dict_path)
keep_tokens = [bert_token_dict[c] for c in chars]


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                ' '.join(text), maxlen=maxlen // self.n + 1
            )
            batch_token_ids.append([0] + token_ids[1:-1])
            batch_segment_ids.append([0] + segment_ids[1:-1])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []
                self.count += 1

    @property
    def n(self):
        if not hasattr(self, 'count'):
            self.count = 0
        if self.count < 20000:
            n = 8
        elif self.count < 40000:
            n = 4
        elif self.count < 80000:
            n = 2
        else:
            n = 1
        return n


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        if mask[1] is None:
            y_mask = 1.0
        else:
            y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='lm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class ChessPlayer(object):
    """交互式下棋程序
    """
    def move_to_chinese(self, move):
        """将单步走法转为中文描述
        """
        if not isinstance(move, Move):
            move = Move(self.board, move[0], move[1])
        return move.to_chinese()

    def move_to_iccs(self, move):
        """将单步走法转为iccs表示
        """
        if not isinstance(move, Move):
            move = Move(self.board, move[0], move[1])
        return move.to_iccs()

    def print_board(self):
        """打印当前棋盘
        直观起见，红方用红色表示，黑方用绿色表示。
        """
        for l in self.board.dump_board():
            for c in u'兵炮车马相仕帅':
                l = l.replace(c, u'\033[1;31;40m%s\033[0m' % c)
            for c in u'卒砲砗碼象士将':
                l = l.replace(c, u'\033[1;32;40m%s\033[0m' % c)
            print(l)

    def movable_steps(self):
        """给出当前局面所有候选走法
        """
        return [self.move_to_iccs(m) for m in self.board.create_moves()]

    def human_input(self):
        """人类行棋
        """
        while True:
            try:
                iccs = input(u'请输入iccs棋着: ')
                print(iccs)
                move = self.board.move_iccs(iccs)
                if move is not None:
                    return iccs, move
            except KeyboardInterrupt:
                return None
            except:
                pass

    def record(self, iccs):
        """将局面往前推进一步
        """
        self.history += iccs
        self.board.next_turn()
        self.print_board()
        self.current = (self.current + 1) % 2

    def new_game(self, current=0):
        """开新局
        """
        self.board = ChessBoard()
        self.board.from_fen(FULL_INIT_FEN)
        self.print_board()
        self.history = ''
        self.current = current
        if self.current == 0:  # 人类先手
            iccs, move = self.human_input()
            self.record(iccs)
        while True:
            # 机器走棋
            moves = self.movable_steps()
            iccses = [' '.join(self.history + m) for m in moves]
            token_ids = [[0] + tokenizer.encode(ic)[0][1:-1] for ic in iccses]
            token_ids = np.array(token_ids)
            segment_ids = np.zeros_like(token_ids)
            preds = model.predict([token_ids, segment_ids])[:, -5:-1]
            preds = np.take_along_axis(preds, token_ids[:, -4:, None], axis=2)
            preds = np.log(preds + 1e-8)[:, :, 0].sum(axis=1)
            iccs = moves[preds.argmax()]
            move = self.board.move_iccs(iccs)
            self.record(iccs)
            if self.board.is_win():
                print(u'机器赢了')
                break
            # 人类走棋
            iccs, move = self.human_input()
            self.record(iccs)
            if self.board.is_win():
                print(u'人类赢了')
                break


chessplayer = ChessPlayer()
"""
chessplayer.new_game(0)  # 启动新棋局，0为人类先手，1为机器先手
"""


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def on_epoch_end(self, epoch, logs=None):
        # 保存模型
        model.save_weights('./best_model.weights')


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')
