# coding:utf-8
import torch
import os
from transformers import *  # 导入所有包
from transformers import BertModel  # 导入指定的包

# transformers的安装
# pip install transformers

# 加载预训练权重和词表
uncased = 'bert-base-uncased'
bert = BertModel.from_pretrained(uncased)
'''
注意：加载预训练权重时需要下载好的预训练的权重文件，一般来说，当缓存文件中没有所需要的文件(第一次使用)，
只要网络没有问题，就会自动下载。当网络出现问题的时候，就需要手动下载预训练权重了。
'''

# 下面的代码需要将bert-base-uncased文件夹与本程序代码在同一个目录下面
UNCASED = './bert-base-uncased'
VOCAB = 'vocab.txt'
tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
bert = BertModel.from_pretrained(UNCASED)
