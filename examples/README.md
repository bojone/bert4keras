# 例子合集

提示：Github上的examples只保证兼容Github上的最新版bert4keras，如果报错，请首先尝试升级bert4keras。

## 简介

- [basic_extract_features.py](https://github.com/bojone/bert4keras/tree/master/examples/basic_extract_features.py): 基础测试，测试BERT对句子的编码序列。
- [basic_gibbs_sampling_via_mlm.py](https://github.com/bojone/bert4keras/tree/master/examples/basic_gibbs_sampling_via_mlm.py): 基础测试，利用BERT+Gibbs采样进行文本随机生成，参考[这里](https://kexue.fm/archives/8119)。
- [basic_language_model_cpm_lm.py](https://github.com/bojone/bert4keras/tree/master/examples/basic_language_model_cpm_lm.py): 基础测试，测试[CPM_LM](https://github.com/TsinghuaAI/CPM-Generate)的生成效果。
- [basic_language_model_gpt2_ml.py](https://github.com/bojone/bert4keras/tree/master/examples/basic_language_model_gpt2_ml.py): 基础测试，测试[GPT2_ML](https://github.com/imcaspar/gpt2-ml)的生成效果。
- [basic_language_model_nezha_gen_gpt.py](https://github.com/bojone/bert4keras/tree/master/examples/basic_language_model_nezha_gen_gpt.py): 基础测试，测试[GPT Base（又叫NEZHE-GEN）](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow)的生成效果。
- [basic_make_uncased_model_cased.py](https://github.com/bojone/bert4keras/tree/master/examples/basic_make_uncased_model_cased.py): 基础测试，通过简单修改词表，使得不区分大小写的模型有区分大小写的能力。
- [basic_masked_language_model.py](https://github.com/bojone/bert4keras/tree/master/examples/basic_masked_language_model.py): 基础测试，测试BERT的MLM模型效果。
- [basic_simple_web_serving_simbert.py](https://github.com/bojone/bert4keras/tree/master/examples/basic_simple_web_serving_simbert.py): 基础测试，测试自带的WebServing（将模型转化为Web接口）。
- [task_conditional_language_model.py](https://github.com/bojone/bert4keras/tree/master/examples/task_conditional_language_model.py): 任务例子，结合 BERT + [Conditional Layer Normalization](https://kexue.fm/archives/7124) 做条件语言模型。
- [task_iflytek_adversarial_training.py](https://github.com/bojone/bert4keras/tree/master/examples/task_iflytek_adversarial_training.py): 任务例子，通过[对抗训练](https://kexue.fm/archives/7234)提升分类效果。
- [task_iflytek_bert_of_theseus.py](https://github.com/bojone/bert4keras/tree/master/examples/task_iflytek_bert_of_theseus.py): 任务例子，通过[BERT-of-Theseus](https://kexue.fm/archives/7575)来进行模型压缩。
- [task_iflytek_gradient_penalty.py](https://github.com/bojone/bert4keras/tree/master/examples/task_iflytek_gradient_penalty.py): 任务例子，通过[梯度惩罚](https://kexue.fm/archives/7234)提升分类效果，可以视为另一种对抗训练。
- [task_iflytek_multigpu.py](https://github.com/bojone/bert4keras/tree/master/examples/task_iflytek_multigpu.py): 任务例子，文本分类多GPU版。
- [task_image_caption.py](https://github.com/bojone/bert4keras/tree/master/examples/task_image_caption.py): 任务例子，BERT + [Conditional Layer Normalization](https://kexue.fm/archives/7124) + ImageNet预训练模型 来做图像描述生成。
- [task_language_model.py](https://github.com/bojone/bert4keras/tree/master/examples/task_language_model.py): 任务例子，加载BERT的预训练权重做无条件语言模型，效果上等价于GPT。
- [task_language_model_chinese_chess.py](https://github.com/bojone/bert4keras/tree/master/examples/task_language_model_chinese_chess.py): 任务例子，用GPT的方式下中国象棋，过程请参考[博客](https://kexue.fm/archives/7877)。
- [task_question_answer_generation_by_seq2seq.py](https://github.com/bojone/bert4keras/tree/master/examples/task_question_answer_generation_by_seq2seq.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做[问答对自动构建](https://kexue.fm/archives/7630)，属于自回归文本生成。
- [task_reading_comprehension_by_mlm.py](https://github.com/bojone/bert4keras/tree/master/examples/task_reading_comprehension_by_mlm.py): 任务例子，通过MLM模型来做[阅读理解问答](https://kexue.fm/archives/7148)，属于简单的非自回归文本生成。
- [task_reading_comprehension_by_seq2seq.py](https://github.com/bojone/bert4keras/tree/master/examples/task_reading_comprehension_by_seq2seq.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做[阅读理解问答](https://kexue.fm/archives/7115)，属于自回归文本生成。
- [task_relation_extraction.py](https://github.com/bojone/bert4keras/tree/master/examples/task_relation_extraction.py): 任务例子，结合BERT以及自行设计的“半指针-半标注”结构来做[关系抽取](https://kexue.fm/archives/7161)。
- [task_sentence_similarity_lcqmc.py](https://github.com/bojone/bert4keras/tree/master/examples/task_sentence_similarity_lcqmc.py): 任务例子，句子对分类任务。
- [task_sentiment_albert.py](https://github.com/bojone/bert4keras/tree/master/examples/task_sentiment_albert.py): 任务例子，情感分类任务，加载ALBERT模型。
- [task_sentiment_integrated_gradients.py](https://github.com/bojone/bert4keras/tree/master/examples/task_sentiment_integrated_gradients.py): 任务例子，通过[积分梯度](https://kexue.fm/archives/7533)的方式可视化情感分类任务。
- [task_sentiment_virtual_adversarial_training.py](https://github.com/bojone/bert4keras/tree/master/examples/task_sentiment_virtual_adversarial_training.py): 任务例子，通过[虚拟对抗训练](https://kexue.fm/archives/7466)进行半监督学习，提升小样本下的情感分类性能。
- [task_seq2seq_ape210k_math_word_problem.py](https://github.com/bojone/bert4keras/tree/master/examples/task_seq2seq_ape210k_math_word_problem.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做小学数学应用题（数学公式生成），详情请见[这里](https://kexue.fm/archives/7809)。
- [task_seq2seq_autotitle.py](https://github.com/bojone/bert4keras/tree/master/examples/task_seq2seq_autotitle.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做新闻标题生成。
- [task_seq2seq_autotitle_csl.py](https://github.com/bojone/bert4keras/tree/master/examples/task_seq2seq_autotitle_csl.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做论文标题生成，包含了评测代码。
- [task_seq2seq_autotitle_csl_mt5.py](https://github.com/bojone/bert4keras/tree/master/examples/task_seq2seq_autotitle_csl_mt5.py): 任务例子，通过[多国语言版T5](https://kexue.fm/archives/7867)式的Seq2Seq模型来做论文标题生成，包含了评测代码。
- [task_seq2seq_autotitle_multigpu.py](https://github.com/bojone/bert4keras/tree/master/examples/task_seq2seq_autotitle_multigpu.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做新闻标题生成，单机多卡版本。
- [task_sequence_labeling_cws_crf.py](https://github.com/bojone/bert4keras/tree/master/examples/task_sequence_labeling_cws_crf.py): 任务例子，通过 BERT + [CRF](https://kexue.fm/archives/7196) 来做中文分词。
- [task_sequence_labeling_ner_crf.py](https://github.com/bojone/bert4keras/tree/master/examples/task_sequence_labeling_ner_crf.py): 
任务例子，通过 BERT + [CRF](https://kexue.fm/archives/7196) 来做中文NER。
