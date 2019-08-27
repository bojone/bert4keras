# bert4keras
- Our light reimplement of bert for keras
- 更清晰、更轻量级的keras版bert
- 个人博客：https://kexue.fm/

## 说明
这是笔者重新实现的keras版的bert，致力于用尽可能少的代码来实现keras下调用bert。

目前已经基本实现bert，并且能成功加载官方权重，经验证模型输出跟keras-bert一致，大家可以放心使用。

本项目的初衷是为了修改、定制上的方便，所以可能会频繁更新。

因此欢迎star，但不建议fork，因为你fork下来的版本可能很快就过期了。

## 使用
参考<a href="https://github.com/bojone/bert4keras/blob/master/test.py">test.py</a>。

之前基于keras-bert给出的<a href="https://github.com/bojone/bert_in_keras">例子</a>，仍适用于本项目，只需要将base_model的加载方式换成本项目的。

目前只保证支持Python 2.7，实验环境是Tesorflow 1.8+以及Keras 2.2.4+。

当然，乐于贡献的朋友如果发现了某些bug的话，也欢迎指出修正甚至Pull Requests～

## 背景
之前一直用CyberZHG大佬的<a href="https://github.com/CyberZHG/keras-bert">keras-bert</a>，如果纯粹只是为了在keras下对bert进行调用和fine tune来说，keras-bert已经足够能让人满意了。

然而，如果想要在加载官方预训练权重的基础上，对bert的内部结构进行修改，那么keras-bert就比较难满足我们的需求了，因为keras-bert为了代码的复用性，几乎将每个小模块都封装为了一个单独的库，比如keras-bert依赖于keras-transformer，而keras-transformer依赖于keras-multi-head，keras-multi-head依赖于keras-self-attention，这样一重重依赖下去，改起来就相当头疼了。

所以，我决定重新写一个keras版的bert，争取在几个文件内把它完整地实现出来，减少这些依赖性，并且保留可以加载官方预训练权重的特性。

## 鸣谢
感谢CyberZHG大佬实现的<a href="https://github.com/CyberZHG/keras-bert">keras-bert</a>，本实现有不少地方参考了keras-bert的源码，在此衷心感谢大佬的无私奉献。
