# bert4keras
- Our light reimplement of bert for keras
- 更清晰、更轻量级的keras版bert
- 个人博客：https://kexue.fm/

## 说明
这是笔者重新实现的keras版的bert，致力于用尽可能清爽的代码来实现keras下调用bert。

目前已经基本实现bert，并且能成功加载官方权重，经验证模型输出跟keras-bert一致，大家可以放心使用。

本项目的初衷是为了修改、定制上的方便，所以可能会频繁更新。

因此欢迎star，但不建议fork，因为你fork下来的版本可能很快就过期了。

## 使用
快速安装：
```shell
pip install git+https://www.github.com/bojone/bert4keras.git
```

使用例子请参考<a href="https://github.com/bojone/bert4keras/blob/master/examples">examples</a>目录。

之前基于keras-bert给出的<a href="https://github.com/bojone/bert_in_keras">例子</a>，仍适用于本项目，只需要将`bert_model`的加载方式换成本项目的。

理论上兼容Python2和Python3，实验环境是Python 2.7、Tesorflow 1.13+以及Keras 2.3.1（已经在2.3.0、2.3.1、tf.keras下测试通过）。

当然，乐于贡献的朋友如果发现了某些bug的话，也欢迎指出修正甚至Pull Requests～

## 权重

目前支持加载的权重：
- <strong>Google原版bert</strong>: https://github.com/google-research/bert
- <strong>徐亮版roberta</strong>: https://github.com/brightmart/roberta_zh
- <strong>哈工大版roberta</strong>: https://github.com/ymcui/Chinese-BERT-wwm
- <strong>Google原版albert</strong><sup><a href="https://github.com/bojone/bert4keras/issues/29#issuecomment-552188981">[例子]</a></sup>: https://github.com/google-research/google-research/tree/master/albert
- <strong>徐亮版albert</strong>: https://github.com/brightmart/albert_zh

（注：徐亮版albert的开源时间早于Google版albert，这导致早期徐亮版albert的权重与Google版的不完全一致，换言之两者不能直接相互替换。为了减少代码冗余，bert4keras的0.2.4及后续版本均只支持加载<u>Google版</u>以徐亮版中<u>带Google字眼</u>的权重。如果要加载早期版本的权重，请用<a href="https://github.com/bojone/bert4keras/releases/tag/v0.2.3">0.2.3版本</a>。）

## 更新
- <strong>2019.11.19</strong>: 将utils.py重命名为tokenizer.py。
- <strong>2019.11.19</strong>: 想来想去，最后还是决定把snippets放到<a href="https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py">bert4keras.snippets</a>下面去好了。
- <strong>2019.11.18</strong>: 优化预训练权重加载逻辑，增加保存模型权重至Bert的checkpoint格式方法。
- <strong>2019.11.17</strong>: <del>分离一些与Bert本身不直接相关的常用代码片段到<a href="https://github.com/bojone/python-snippets">python_snippets</a>，供其它项目共用。</del>
- <strong>2019.11.11</strong>: 添加NSP部分。
- <strong>2019.11.05</strong>: 适配<a href="https://github.com/google-research/google-research/tree/master/albert">google版albert</a>，不再支持<a href="https://github.com/brightmart/albert_zh">非Google版albert_zh</a>。
- <strong>2019.11.05</strong>: 以RoBERTa为例子的预训练代码开发完毕，同时支持TPU/多GPU训练，详见<a href="https://github.com/bojone/bert4keras/tree/master/pretraining/roberta/">roberta</a>。欢迎在此基础上构建更多的预训练代码。
- <strong>2019.11.01</strong>: 逐步增加预训练相关代码，详见<a href="https://github.com/bojone/bert4keras/tree/master/pretraining">pretraining</a>。
- <strong>2019.10.28</strong>: 支持使用基于<a href="https://github.com/google/sentencepiece">sentencepiece</a>的tokenizer。
- <strong>2019.10.25</strong>: 引入原生tokenizer。
- <strong>2019.10.22</strong>: 引入梯度累积优化器。
- <strong>2019.10.21</strong>: 为了简化代码结构，决定放弃keras 2.3.0之前的版本的支持，目前只支持keras 2.3.0+以及tf.keras。
- <strong>2019.10.20</strong>: 应网友要求，现支持直接用`model.save`保存模型结构，用`load_model`加载整个模型（只需要在`load_model`之前执行`from bert4keras.layers import *`，不需要额外写`custom_objects`）。
- <strong>2019.10.09</strong>: 已兼容tf.keras，同时在tf 1.13和tf 2.0下的tf.keras测试通过，通过设置环境变量`TF_KERAS=1`来切换tf.keras。
- <strong>2019.10.09</strong>: 已兼容Keras 2.3.x，但只是临时方案，后续可能直接移除掉2.3之前版本的支持。
- <strong>2019.10.02</strong>: 适配albert，能成功加载<a href="https://github.com/brightmart/albert_zh">albert_zh</a>的权重，只需要在`load_pretrained_model`函数里加上`albert=True`。

## 背景
之前一直用CyberZHG大佬的<a href="https://github.com/CyberZHG/keras-bert">keras-bert</a>，如果纯粹只是为了在keras下对bert进行调用和fine tune来说，keras-bert已经足够能让人满意了。

然而，如果想要在加载官方预训练权重的基础上，对bert的内部结构进行修改，那么keras-bert就比较难满足我们的需求了，因为keras-bert为了代码的复用性，几乎将每个小模块都封装为了一个单独的库，比如keras-bert依赖于keras-transformer，而keras-transformer依赖于keras-multi-head，keras-multi-head依赖于keras-self-attention，这样一重重依赖下去，改起来就相当头疼了。

所以，我决定重新写一个keras版的bert，争取在几个文件内把它完整地实现出来，减少这些依赖性，并且保留可以加载官方预训练权重的特性。

## 鸣谢
感谢CyberZHG大佬实现的<a href="https://github.com/CyberZHG/keras-bert">keras-bert</a>，本实现有不少地方参考了keras-bert的源码，在此衷心感谢大佬的无私奉献。

## 交流
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn
