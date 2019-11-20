#! -*- coding: utf-8 -*-
# bert做language model任务，小说生成

from __future__ import print_function
import glob
import numpy as np
from tqdm import tqdm
import os, json, codecs, re
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer, load_vocab
from bert4keras.snippets import sequence_padding
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


lm_config = 'lm_config.json'
min_count = 8
maxlen = 256
batch_size = 16
steps_per_epoch = 1000
epochs = 10000

# bert配置
config_path = '../../bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


novels = []

for txt in glob.glob('../金庸/*/*.txt'):
    txt = open(txt).read()
    txt = txt.decode('gbk', 'ignore')
    txt = txt.replace('\r', '').replace('\n', '')
    txt = txt.replace(u'整理制作，并提供下载', '')
    txt = re.sub(u'www.*?com', '', txt)
    txt = txt.replace(u'\u3000', ' ')
    sents = []
    for t in txt.split('  '):
        for s in re.findall(u'.*?。', t):
            if len(s) <= maxlen - 2:
                sents.append(s)
    novels.append(sents)


_token_dict = load_vocab(dict_path) # 读取词典
_tokenizer = Tokenizer(_token_dict) # 建立临时分词器

if os.path.exists(lm_config):
    tokens = json.load(open(lm_config))
else:
    tokens = {}
    for novel in novels:
        for s in novel:
            for t in _tokenizer.tokenize(s):
                tokens[t] = tokens.get(t, 0) + 1
    tokens = [(i, j) for i, j in tokens.items() if j >= min_count]
    tokens = sorted(tokens, key=lambda t: -t[1])
    tokens = [t[0] for t in tokens]
    json.dump(tokens,
              codecs.open(lm_config, 'w', encoding='utf-8'),
              indent=4,
              ensure_ascii=False)

token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_dict[t] = len(token_dict)
    keep_words.append(_token_dict[t])

for t in tokens:
    if t in _token_dict and t not in token_dict:
        token_dict[t] = len(token_dict)
        keep_words.append(_token_dict[t])

tokenizer = Tokenizer(token_dict)  # 建立分词器


data = []
pbar = tqdm(desc=u'构建语料中', total=sum(len(n) for n in novels))

for novel in novels:
    s = u''
    for i in range(len(novel)):
        for j in range(len(novel) - i):
            if len(s) + len(novel[i + j]) > maxlen - 2:
                data.append(s)
                s = u''
                break
            else:
                s += novel[i + j]
        pbar.update(1)
        if i + j >= len(novel):
            break
    if s:
        data.append(s)

pbar.close()
np.random.shuffle(data)


def data_generator():
    while True:
        X, S = [], []
        for d in data:
            x, s = tokenizer.encode(d)
            X.append(x)
            S.append(s)
            if len(X) == batch_size:
                X = sequence_padding(X)
                S = sequence_padding(S)
                yield [X, S], None
                X, S = [], []


model = build_bert_model(
    config_path,
    checkpoint_path,
    application='lm',
    keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
)

model.summary()

# 交叉熵作为loss，并mask掉输入部分的预测
y_in = model.input[0][:, 1:]  # 目标tokens
y_mask = model.get_layer('Sequence-Mask').output[:, 1:] # 目标mask
y = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))


def random_generate(s, n=1, topk=5):
    """随机采样生成
    每次从最高概率的topk个token中随机采样一个
    """
    token_ids, segment_ids = tokenizer.encode(s)
    token_ids, segment_ids = token_ids[: -1], segment_ids[: -1]
    target_ids = [[] for _ in range(n)]
    R = []
    for i in range(maxlen):
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [0] * len(t) for t in target_ids]
        # 下面直接忽略[PAD], [UNK], [CLS]
        _probas = model.predict([_target_ids, _segment_ids])[:, -1, 3:]
        for i, p in enumerate(_probas):
            p_arg_topk = p.argsort()[::-1][:topk]
            p_topk = p[p_arg_topk]
            p = p_topk / sum(p_topk)
            idx = np.random.choice(len(p), p=p)
            target_ids[i].append(p_arg_topk[idx] + 3)
        for t in target_ids:
            if t[-1] == 3:
                R.append(tokenizer.decode(t))
        target_ids = [t for t in target_ids if t[-1] != 3]
        if len(target_ids) == 0:
            break
    return R


def just_show():
    s1 = u'当晚两人在一家小客店中宿歇。张无忌躺在炕上，越想越是担心，走到赵敏窗外，但听她呼吸调匀，正自香梦沉酣。'
    s2 = u'虚竹飞身跃上松树的枝干，只见段延庆的钢杖深深嵌在树枝之中，全凭一股内力粘劲，挂住了下面四人，内力之深厚，实是非同小可。虚竹伸左手抓住钢杖，提将上来。'
    s3 = u'杨过居住在侠客岛，是令狐冲的弟子，武器是金蛇剑。'
    for s in [s1, s2, s3]:
        t = [s + i for i in random_generate(s)]
        print(u'输入: %s' % s)
        print(u'结果: %s\n' % ('\n'.join(t)))


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        # 演示效果
        just_show()


if __name__ == '__main__':

    evaluator = Evaluate()

    model.fit_generator(data_generator(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=[evaluator])

else:

    model.load_weights('./best_model.weights')
    
"""
效果：

输入: 当晚两人在一家小客店中宿歇。张无忌躺在炕上，越想越是担心，走到赵敏窗外，但听她呼吸调匀，正自香梦沉酣。
结果: 当晚两人在一家小客店中宿歇。张无忌躺在炕上，越想越是担心，走到赵敏窗外，但听她呼吸调匀，正自香梦沉酣。次日清晨，张无忌便和赵敏去买了一匹高头大马，自己骑了随伴。那马甚有神骏，三十六斤重的身躯之中，竟无一头白马。他心中怦怦乱跳，暗想：若能将赵敏引出迷城，我决不致再和她相会，但若和赵姑娘相遇，我一生一世决计再难相见。何况我是她的私生女儿，这般亲热，岂不是好？我如何能和她相见？今后我要教训教训她才好？我教教她，教训她，要她心里快快活活的。他心如刀割，当即回到客店，将张无忌的所在说了。

输入: 虚竹飞身跃上松树的枝干，只见段延庆的钢杖深深嵌在树枝之中，全凭一股内力粘劲，挂住了下面四人，内力之深厚，实是非同小可。虚竹伸左手抓住钢杖，提将上来。
结果: 虚竹飞身跃上松树的枝干，只见段延庆的钢杖深深嵌在树枝之中，全凭一股内力粘劲，挂住了下面四人，内力之深厚，实是非同小可。虚竹伸左手抓住钢杖，提将上来。那矮子见他如此功力，大吃一惊，叫道：什么人？是谁？你干什么？我师父是谁？你们是谁？是谁？你们是谁？我师父是谁？你这矮子，便是段延庆。你们不知道我师父便是，是不是？快快说来。那矮子道：我师父便是延庆太子，他的徒弟也是段延庆。他老人家在唐朝做镇南王，你们便将他改名为延庆太子，叫做延庆太子！这名头倒怪，你们大伙儿听见了，也不知道他老人家是死是活。

输入: 杨过居住在侠客岛，是令狐冲的弟子，武器是金蛇剑。
结果: 杨过居住在侠客岛，是令狐冲的弟子，武器是金蛇剑。这时见他手中所握，竟是一柄特制的短剑，心中大喜，叫道：：原来是金蛇郎君的剑！原来你便是金蛇郎君的弟子，这一下可要叫我失望了。那人哈哈一笑，说道：好啊！好啊，好啊！我的金蛇剑是我的，不过我是你的。这人道：我姓杨名过，名字叫过。你是我儿子，是我女儿，是不是？你这么大的年纪，怎地自称金刀驸马？我这就给你取个名字，叫作过儿。
"""
