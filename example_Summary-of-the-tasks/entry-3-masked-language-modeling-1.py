# coding:utf-8
from transformers import pipeline
from pprint import pprint

# here is an example of using pipelines to replace a mask from a sequence
nlp = pipeline("fill-mask")

pprint(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))

'''
第一次运行同样需要进行模型的下载
输出结果如下所示：

'''