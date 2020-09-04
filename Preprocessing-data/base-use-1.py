# coding:utf-8
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

encoded_input = tokenizer('Hello, I am a single sentence!')
print(encoded_input)

'''
输出的结果如下所示：
{'input_ids': [101, 8667, 117, 146, 1821, 170, 1423, 5650, 106, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

'''

print(tokenizer.decode(encoded_input['input_ids']))

''''
输出的解雇如下所示：
[CLS] Hello, I am a single sentence! [SEP]
'''