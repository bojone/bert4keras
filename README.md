# bert4keras
- Our light reimplement of bert for keras
- 更清晰、更轻量级的keras版bert
- 个人博客：https://kexue.fm/
- 在线文档：http://bert4keras.spaces.ac.cn/ （还在构建中）

## 说明
这是笔者重新实现的keras版的transformer模型库，致力于用尽可能清爽的代码来实现结合transformer和keras。

本项目的初衷是为了修改、定制上的方便，所以可能会频繁更新。

因此欢迎star，但不建议fork，因为你fork下来的版本可能很快就过期了。

## 功能
目前已经实现：
- 加载bert/roberta/albert的预训练权重进行finetune；
- 实现语言模型、seq2seq所需要的attention mask；
- 丰富的<a href="https://github.com/bojone/bert4keras/tree/master/examples">examples</a>；
- 从零预训练代码（支持TPU、多GPU，请看<a href="https://github.com/bojone/bert4keras/tree/master/pretraining">pretraining</a>）；
- 兼容keras、tf.keras

## 使用
安装稳定版：
```shell
pip install bert4keras
```
安装最新版：
```shell
pip install git+https://www.github.com/bojone/bert4keras.git
```

使用例子请参考<a href="https://github.com/bojone/bert4keras/blob/master/examples">examples</a>目录。

之前基于keras-bert给出的<a href="https://github.com/bojone/bert_in_keras">例子</a>，仍适用于本项目，只需要将`bert_model`的加载方式换成本项目的。

理论上兼容Python2和Python3，兼容tensorflow 1.14+和tensorflow 2.x，实验环境是Python 2.7、Tesorflow 1.14+以及Keras 2.3.1（已经在2.2.4、2.3.0、2.3.1、tf.keras下测试通过）。

**为了获得最好的体验，建议你使用Tensorflow 1.14 + Keras 2.3.1组合。**

<blockquote><strong>关于环境组合</strong>
  
- 支持tf+keras和tf+tf.keras，后者需要提前传入环境变量TF_KERAS=1。

- 当使用tf+keras时，建议2.2.4 <= keras <= 2.3.1，以及 1.14 <= tf <= 2.2，不能使用tf 2.3+。

- keras 2.4+可以用，但事实上keras 2.4.x基本上已经完全等价于tf.keras了，因此如果你要用keras 2.4+，倒不如直接用tf.keras。
</blockquote>

当然，乐于贡献的朋友如果发现了某些bug的话，也欢迎指出修正甚至Pull Requests～

## 权重

目前支持加载的权重：
- <strong>Google原版bert</strong>: https://github.com/google-research/bert
- <strong>brightmart版roberta</strong>: https://github.com/brightmart/roberta_zh
- <strong>哈工大版roberta</strong>: https://github.com/ymcui/Chinese-BERT-wwm
- <strong>Google原版albert</strong><sup><a href="https://github.com/bojone/bert4keras/issues/29#issuecomment-552188981">[例子]</a></sup>: https://github.com/google-research/ALBERT
- <strong>brightmart版albert</strong>: https://github.com/brightmart/albert_zh
- <strong>转换后的albert</strong>: https://github.com/bojone/albert_zh
- <strong>华为的NEZHA</strong>: https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow
- <strong>华为的NEZHA-GEN</strong>: https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow
- <strong>自研语言模型</strong>: https://github.com/ZhuiyiTechnology/pretrained-models
- <strong>T5模型</strong>: https://github.com/google-research/text-to-text-transfer-transformer
- <strong>GPT_OpenAI</strong>: https://github.com/bojone/CDial-GPT-tf
- <strong>GPT2_ML</strong>: https://github.com/imcaspar/gpt2-ml
- <strong>Google原版ELECTRA</strong>: https://github.com/google-research/electra
- <strong>哈工大版ELECTRA</strong>: https://github.com/ymcui/Chinese-ELECTRA
- <strong>CLUE版ELECTRA</strong>: https://github.com/CLUEbenchmark/ELECTRA
- <strong>LaBSE（多国语言BERT）</strong>: https://github.com/bojone/labse
- <strong>Chinese-GEN项目下的模型</strong>: https://github.com/bojone/chinese-gen
- <strong>T5.1.1</strong>: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md#t511
- <strong>Multilingual T5</strong>: https://github.com/google-research/multilingual-t5/

<strong>注意事项</strong>
- 注1：brightmart版albert的开源时间早于Google版albert，这导致早期brightmart版albert的权重与Google版的不完全一致，换言之两者不能直接相互替换。为了减少代码冗余，bert4keras的0.2.4及后续版本均只支持加载<u>Google版</u>以brightmart版中<u>带Google字眼</u>的权重。如果要加载早期版本的权重，请用<a href="https://github.com/bojone/bert4keras/releases/tag/v0.2.3">0.2.3版本</a>，或者考虑作者转换过的<a href="https://github.com/bojone/albert_zh">albert_zh</a>。
- 注2：下载下来的ELECTRA权重，如果没有json配置文件的话，参考<a href="https://github.com/ymcui/Chinese-ELECTRA/issues/3">这里</a>自己改一个（需要加上`type_vocab_size`字段）。

## 更新
- <strong>2023.03.06</strong>: [无穷大改np.inf；优化显存占用](https://github.com/bojone/bert4keras/commit/20a46946156b4bc15ceaa00671fcd00c8b702640)。将无穷大改为np.inf，运算更加准确，而且在低精度运算时不容易出错；同时合并了若干mask算子，减少了显存占用。实测在A100上训练base和large级别模型时，速度有明显加快，显存占用也有降低。
- <strong>2022.03.20</strong>: 增加[RoFormerV2](https://kexue.fm/archives/8998)。
- <strong>2022.02.28</strong>: 增加[GatedAttentionUnit](https://kexue.fm/archives/8934)。
- <strong>2021.04.23</strong>: 增加[GlobalPointer](https://kexue.fm/archives/8373)。
- <strong>2021.03.23</strong>: 增加[RoFormer](https://kexue.fm/archives/8265)。
- <strong>2021.01.30</strong>: 发布0.9.9版，完善多GPU支持，增加多GPU例子：[task_seq2seq_autotitle_multigpu.py](https://github.com/bojone/bert4keras/blob/master/examples/task_seq2seq_autotitle_multigpu.py)。
- <strong>2020.12.29</strong>: 增加`residual_attention_scores`参数来实现RealFormer，只需要在`build_transformer_model`中传入参数`residual_attention_scores=True`启用。
- <strong>2020.12.04</strong>: `PositionEmbedding`引入层次分解，可以让BERT直接处理超长文本，在`build_transformer_model`中传入参数`hierarchical_position=True`启用。
- <strong>2020.11.19</strong>: 支持GPT2模型，参考[CPM_LM_bert4keras](https://github.com/bojone/CPM_LM_bert4keras)项目。
- <strong>2020.11.14</strong>: 新增分参数学习率`extend_with_parameter_wise_lr`，可用于给每层设置不同的学习率。
- <strong>2020.10.27</strong>: 支持<a href="https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md#t511">T5.1.1</a>和<a href="https://github.com/google-research/multilingual-t5/">Multilingual T5</a>。
- <strong>2020.08.28</strong>: 支持<a href="https://github.com/bojone/CDial-GPT-tf">GPT_OpenAI</a>。
- <strong>2020.08.22</strong>: 新增`WebServing`类，允许简单地将模型转换为Web接口，详情请参考该类的<a href="https://github.com/bojone/bert4keras/blob/8d55512a12e4677262363ac189ebf504fc451716/bert4keras/snippets.py#L580">说明</a>。
- <strong>2020.07.14</strong>: `Transformer`类加入`prefix`参数；`snippets.py`引入`to_array`函数；`AutoRegressiveDecoder`修改`rtype='logits'`时的一个隐藏bug。
- <strong>2020.06.06</strong>: 强迫症作祟：将`Tokenizer`原来的`max_length`参数重命名为`maxlen`，同时保留向后兼容性，建议大家用新参数名。
- <strong>2020.04.29</strong>: 增加重计算（参考<a href="https://github.com/bojone/keras_recompute">keras_recompute</a>），可以通过时间换空间，通过设置环境变量`RECOMPUTE=1`启用。
- <strong>2020.04.25</strong>: 优化tf2下的表现。
- <strong>2020.04.16</strong>: 所有example均适配tensorflow 2.0。
- <strong>2020.04.06</strong>: 增加UniLM预训练模式（测试中）。
- <strong>2020.04.06</strong>: 完善`rematch`方法。
- <strong>2020.04.01</strong>: `Tokenizer`增加`rematch`方法，给出分词结果与原序列的映射关系。
- <strong>2020.03.30</strong>: 尽量统一py文件的写法。
- <strong>2020.03.25</strong>: 支持ELECTRA。
- <strong>2020.03.24</strong>: 继续加强`DataGenerator`，允许传入迭代器时进行局部shuffle。
- <strong>2020.03.23</strong>: 增加调整Attention的`key_size`的选项。
- <strong>2020.03.17</strong>: 增强`DataGenerator`；优化模型写法。
- <strong>2020.03.15</strong>: 支持<a href="https://github.com/imcaspar/gpt2-ml">GPT2_ML</a>。
- <strong>2020.03.10</strong>: 支持Google的<a href="https://github.com/google-research/text-to-text-transfer-transformer">T5</a>模型。
- <strong>2020.03.05</strong>: 将`tokenizer.py`更名为`tokenizers.py`。
- <strong>2020.03.05</strong>: `application='seq2seq'`改名为`application='unilm'`。
- <strong>2020.03.05</strong>: `build_bert_model`更名为`build_transformer_model`。
- <strong>2020.03.05</strong>: 重写`models.py`结构。
- <strong>2020.03.04</strong>: 将`bert.py`更名为`models.py`。
- <strong>2020.03.02</strong>: 重构mask机制（用回Keras自带的mask机制），以便更好地编写更复杂的应用。
- <strong>2020.02.22</strong>: 新增`AutoRegressiveDecoder`类，统一处理Seq2Seq的解码问题。
- <strong>2020.02.19</strong>: transformer block的前缀改为Transformer（本来是Encoder），使得其含义局限性更少。
- <strong>2020.02.13</strong>: 优化`load_vocab`函数；将`build_bert_model`中的`keep_words`参数更名为`keep_tokens`，此处改动可能会对部分脚本产生影响。
- <strong>2020.01.18</strong>: 调整文本处理方式，去掉codecs的使用。
- <strong>2020.01.17</strong>: 各api日趋稳定，为了方便大家使用，打包到<a href="https://pypi.org/project/bert4keras/">pypi</a>，首个打包版本号为0.4.6。
- <strong>2020.01.10</strong>: 重写模型mask方案，某种程度上让代码更为简练清晰；后端优化。
- <strong>2019.12.27</strong>: 重构预训练代码，减少冗余；目前支持RoBERTa和GPT两种预训练方式，详见<a href="https://github.com/bojone/bert4keras/tree/master/pretraining/">pretraining</a>。
- <strong>2019.12.17</strong>: 适配华为的<a href="https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA">nezha</a>权重，只需要在`build_bert_model`函数里加上`model='nezha'`；此外原来albert的加载方式`albert=True`改为`model='albert'`。
- <strong>2019.12.16</strong>: 通过跟keras 2.3+版本类似的思路给低版本引入层中层功能，从而恢复对低于2.3.0版本的keras的支持。
- <strong>2019.12.14</strong>: 新增Conditional Layer Normalization及相关demo。
- <strong>2019.12.09</strong>: 各example的data_generator规范化；修复application='lm'时的一个错误。
- <strong>2019.12.05</strong>: 优化tokenizer的do_lower_case，同时微调各个example。
- <strong>2019.11.23</strong>: 将train.py重命名为optimizers.py，更新大量优化器实现，全面兼容keras和tf.keras。
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

## 引用

```
@misc{bert4keras,
  title={bert4keras},
  author={Jianlin Su},
  year={2020},
  howpublished={\url{https://bert4keras.spaces.ac.cn}},
}
```

## 交流
QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn
