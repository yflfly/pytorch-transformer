# coding:utf-8
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
a = "This is a short sequence."
b = "This is a rather long sequence. It is at least longer than the sequence A."
encoded_a = tokenizer(a)["input_ids"]
encoded_b = tokenizer(b)["input_ids"]

print(encoded_a)
print(encoded_b)
'''
输出的结果如下所示：
[101, 1188, 1110, 170, 1603, 4954, 119, 102]
[101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]
'''
print(len(encoded_a), len(encoded_b))  # 输出：8 19

# 将句子短的进行padding操作，保证句子的长度一致
padded_sequences = tokenizer([a, b], padding=True)
print(padded_sequences)
'''
返回的结果是一个字典
输出的结果如下所示：
{'input_ids': [[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]], 
'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

'''
