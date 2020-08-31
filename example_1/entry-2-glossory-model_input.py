# coding:utf-8
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence = "A Titan RTX has 24G of VARM"
tokenized_sequence = tokenizer.tokenize(sequence)
print(tokenized_sequence)

'''
输出的结果为：
['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##G', 'of', 'VA', '##R', '##M']
'''

encoded_sequence = tokenizer(sequence)["input_ids"]
print(encoded_sequence)

'''
The tokenizer returns a dictionary with all the arguments necessary for its corresponding model to work properly. The token indices are under the key “input_ids”:
输出的结果为:
[101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]
'''

decoded_sequence = tokenizer.decode(encoded_sequence)
print(decoded_sequence)
'''
这个是一个BERT模型期望输入的方式
输出的结果为：
[CLS] A Titan RTX has 24G of VARM [SEP]
'''