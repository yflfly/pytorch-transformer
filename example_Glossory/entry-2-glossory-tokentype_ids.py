# coding:utf-8
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
a = "This is a short sequence."
b = "This is a rather long sequence. It is at least longer than the sequence A."
encoded_dict = tokenizer(a, b)
decoded = tokenizer.decode((encoded_dict["input_ids"]))

print(decoded)

'''
输出的结果如下所示：
[CLS] This is a short sequence. [SEP] This is a rather long sequence. It is at least longer than the sequence A. [SEP]
'''
print(encoded_dict['token_type_ids'])
'''
输出结果如下所示：
[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
'''