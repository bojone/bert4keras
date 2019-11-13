# RoBERTa模式的预训练代码

可能是最清晰、最通用的bert预训练源码。

## 使用
```
python data_utils.py # 生成tfrecord
python pretraining.py # 启动预训练过程
```

请阅读`data_utils.py`和`pretraining.py`修改相应的配置和参数，以适配自己的语料和设备。
