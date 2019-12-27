# 预训练相关代码

目前支持RoBERTa和GPT模式的预训练。

## 使用
```
python data_utils.py # 生成tfrecord
python pretraining.py # 启动预训练过程
```

请阅读`data_utils.py`和`pretraining.py`修改相应的配置和参数，以适配自己的语料和设备。

## 背景

keras是一个友好的框架，通常我们都是基于tf后端使用，另外还有tf.keras可以使用，基本上跟keras 2.3.x的接口一致了。

这种一致性意味着使用keras几乎就相当于使用tf，这意味着tf的一切优势keras也有，但tf没有的优势（比如使用简便）keras也有。

因此，作者参考原训练过程地实现了基于keras的预训练脚本，而有了这个keras版之后，因为前面所述的一致性，所以我们可以很轻松地迁移到多GPU上训练，也可以很轻松地迁移到TPU上训练。
