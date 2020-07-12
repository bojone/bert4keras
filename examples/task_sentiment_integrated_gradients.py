#! -*- coding: utf-8 -*-
# 通过积分梯度（Integrated Gradients）来给输入进行重要性排序
# 接 task_sentiment_albert.py
# 原始论文：https://arxiv.org/abs/1703.01365
# 博客介绍：https://kexue.fm/archives/7533
# 请读者务必先弄懂原理再看代码，下述代码仅是交互式演示代码，并非成品API

from task_sentiment_albert import *
from keras.layers import Layer, Input
from bert4keras.backend import K, batch_gather
from keras.models import Model
from bert4keras.snippets import uniout


class Gradient(Layer):
    """获取梯度的层
    """
    def __init__(self, **kwargs):
        super(Gradient, self).__init__(**kwargs)
        self.supports_masking = True
    def call(self, input):
        input, output, label = input
        output = batch_gather(output, label)
        return K.gradients(output, [input])[0] * input
    def compute_output_shape(self, input_shape):
        return input_shape[0]


label_in = Input(shape=(1,))  # 指定标签
input = model.get_layer('Embedding-Token').output
output = model.output
grads = Gradient()([input, output, label_in])
grad_model = Model(model.inputs + [label_in], grads)

# 获取原始embedding层
embeddings = model.get_layer('Embedding-Token').embeddings
values = K.eval(embeddings)

text = u'这家店真黑心'
text = u'图太乱了 有点看不懂重点  讲故事的时候很难让孩子集中'
text = u'这是一本很好看的书'
text = u'这是一本很糟糕的书'
token_ids, segment_ids = tokenizer.encode(text)
preds = model.predict([[token_ids], [segment_ids]])
label = np.argmax(preds[0])

pred_grads = []
n = 20
for i in range(n):
    # nlp任务中参照背景通常直接选零向量，所以这里
    # 让embedding层从零渐变到原始值，以实现路径变换。
    alpha = 1.0 * i / (n - 1)
    K.set_value(embeddings, alpha * values)
    pred_grad = grad_model.predict([[token_ids], [segment_ids], [[label]]])[0]
    pred_grads.append(pred_grad)

# 然后求平均
pred_grads = np.mean(pred_grads, 0)

# 这时候我们得到形状为(seq_len, hidden_dim)的矩阵，我们要将它变换成(seq_len,)
# 这时候有两种方案：1、直接求模长；2、取绝对值后再取最大。两者效果差不多。
scores = np.sqrt((pred_grads**2).sum(axis=1))
scores = (scores - scores.min()) / (scores.max() - scores.min())
scores = scores.round(4)
results = [(tokenizer.decode([t]), s) for t, s in zip(token_ids, scores)]
print(results[1:-1])

scores = np.abs(pred_grads).max(axis=1)
scores = (scores - scores.min()) / (scores.max() - scores.min())
scores = scores.round(4)
results = [(tokenizer.decode([t]), s) for t, s in zip(token_ids, scores)]
print(results[1:-1])
