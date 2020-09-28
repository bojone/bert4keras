#! -*- coding: utf-8 -*-
# bert做language model任务，小说生成

from __future__ import print_function
import glob, re
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model

maxlen = 256
batch_size = 16
steps_per_epoch = 1000
epochs = 10000

# bert配置
config_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

novels = []

for txt in glob.glob('/root/金庸/*/*.txt'):
    txt = open(txt, encoding='gbk').read()
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

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

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


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, text in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


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


class StoryCompletion(AutoRegressiveDecoder):
    """基于随机采样的故事续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0]
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.zeros_like(token_ids)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids[:-1]], n, topk)  # 基于随机采样
        return [text + tokenizer.decode(ids) for ids in results]


story_completion = StoryCompletion(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def just_show():
    s1 = u'当晚两人在一家小客店中宿歇。张无忌躺在炕上，越想越是担心，走到赵敏窗外，但听她呼吸调匀，正自香梦沉酣。'
    s2 = u'虚竹飞身跃上松树的枝干，只见段延庆的钢杖深深嵌在树枝之中，全凭一股内力粘劲，挂住了下面四人，内力之深厚，实是非同小可。虚竹伸左手抓住钢杖，提将上来。'
    s3 = u'杨过居住在侠客岛，是令狐冲的弟子，武器是金蛇剑。'
    for s in [s1, s2, s3]:
        t = story_completion.generate(s)
        print(u'输入: %s' % s)
        print(u'结果: %s\n' % ('\n'.join(t)))


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
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

    evaluator = Evaluator()
    train_generator = data_generator(data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

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
