# learn_pytorch

介绍模型的英文网站：

https://huggingface.co/transformers/

代码来源网址：

https://huggingface.co/transformers/quicktour.html

model hub:

https://huggingface.co/models

github 网址：https://github.com/huggingface/transformers

1、 example_1 情感分析任务 运行结果截图如下所示：

![image](https://github.com/yflfly/learn_pytorch/tree/master/pytorch-transformer/image/example_1.png)

## 1、简介
Huggingface总部位于纽约，是一家专注于自然语言处理、人工智能和分布式系统的创业公司

学习的系列网站：

https://zhuanlan.zhihu.com/p/164966421

Transformers的组件和模型架构：

1）Configuration配置类：存储模型和分词器的参数，诸如词表大小，隐层维数，dropout rate等。配置类对深度学习框架是透明的

2）Tokenizer分词器类。每个模型都有对应的分词器，存储token到index的映射，负责每个模型特定的序列编码解码流程，比如BPE(Byte Pair Encoding)，SentencePiece等等。也可以方便地添加特殊token或者调整词表大小，如CLS、SEP等等。

3）Model模型类。提供一个基类，实现模型的计算图和编码过程，实现前向传播过程，通过一系列self-attention层直到最后一个隐藏状态层。在最后一层基础上，根据不同的应用会再做些封装，比如XXXForSequenceClassification，XXXForMaskedLM这些派生类。

写作的demo:

https://transformer.huggingface.co/

