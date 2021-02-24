#! -*- coding: utf-8 -*-
# 基本测试：清华开源的中文GPT2模型（26亿参数）
# 项目链接：https://github.com/TsinghuaAI/CPM-Generate
# 博客介绍：https://kexue.fm/archives/7912

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import SpTokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout
import jieba
jieba.initialize()

# 模型路径
config_path = '/root/kg/bert/CPM_LM_2.6B_TF/config.json'
checkpoint_path = '/root/kg/bert/CPM_LM_2.6B_TF/model.ckpt'
spm_path = '/root/kg/bert/CPM_LM_2.6B_TF/chinese_vocab.model'


def pre_tokenize(text):
    """分词前处理函数
    """
    return [
        w.replace(' ', u'\u2582').replace('\n', u'\u2583')
        for w in jieba.cut(text, cut_all=False)
    ]


tokenizer = SpTokenizer(
    spm_path,
    token_start=None,
    token_end=None,
    pre_tokenize=pre_tokenize,
    token_translate={u'\u2583': '<cls>'}
)  # 建立分词器

model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, model='gpt2'
)  # 建立模型，加载权重


class TextExpansion(AutoRegressiveDecoder):
    """基于随机采样的文本续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = np.concatenate([inputs[0], output_ids], 1)
        return self.last_token(model).predict(token_ids)

    def generate(self, text, n=1, topp=0.95, temperature=1):
        """输出结果会有一定的随机性，如果只关心Few Shot效果，
        可以考虑将解码方式换为beam search。
        """
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids],
                                     n,
                                     topp=topp,
                                     temperature=temperature)  # 基于随机采样
        results = [token_ids + [int(i) for i in ids] for ids in results]
        texts = [tokenizer.decode(ids) for ids in results]
        return [self.post_replace(text) for text in texts]

    def post_replace(self, text):
        for s, t in [(' ', ''), (u'\u2582', ' '), (u'\u2583', '\n')]:
            text = text.replace(s, t)
        return text


text_expansion = TextExpansion(
    start_id=None,
    end_id=3,  # 3是<cls>，也是换行符
    maxlen=16,
)

# 常识推理
# 本例输出：北京
query = u"""
美国的首都是华盛顿
法国的首都是巴黎
日本的首都是东京
中国的首都是
"""
print(text_expansion.generate(query[1:-1], 1)[0])

# 单词翻译
# 本例输出：bird
query = u"""
狗 dog
猫 cat
猪 pig
鸟 
"""
print(text_expansion.generate(query[1:-1], 1)[0])

# 主语抽取
# 本例输出：杨振宁
query = u"""
从1931年起，华罗庚在清华大学边学习边工作 华罗庚
在一间简陋的房间里，陈景润攻克了“哥德巴赫猜想” 陈景润
在这里，丘成桐得到IBM奖学金 丘成桐
杨振宁在粒子物理学、统计力学和凝聚态物理等领域作出里程碑性贡献 
"""
print(text_expansion.generate(query[1:-1], 1)[0])

# 三元组抽取
# 本例输出：张红,体重,140斤
query = u"""
姚明的身高是211cm，是很多人心目中的偶像。 ->姚明，身高，211cm
毛泽东是绍兴人，早年在长沙读书。->毛泽东，出生地，绍兴
虽然周杰伦在欧洲办的婚礼，但是他是土生土长的中国人->周杰伦，国籍，中国
小明出生于武汉，但是却不喜欢在武汉生成，长大后去了北京。->小明，出生地，武汉
吴亦凡是很多人的偶像，但是他却是加拿大人，另很多人失望->吴亦凡，国籍，加拿大
武耀的生日在5月8号，这一天，大家都为他庆祝了生日->武耀，生日，5月8号
《青花瓷》是周杰伦最得意的一首歌。->周杰伦，作品，《青花瓷》
北京是中国的首都。->中国，首都，北京
蒋碧的家乡在盘龙城，毕业后去了深圳工作。->蒋碧，籍贯，盘龙城
上周我们和王立一起去了他的家乡云南玩昨天才回到了武汉。->王立，籍贯，云南
昨天11月17号，我和朋友一起去了海底捞，期间服务员为我的朋友刘章庆祝了生日。->刘章，生日，11月17号
张红的体重达到了140斤，她很苦恼。->
"""
print(text_expansion.generate(query[1:-1], 1)[0])
