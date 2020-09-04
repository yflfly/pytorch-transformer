# coding:utf-8
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# 处理句子对
encoded_input = tokenizer("How old are you?", "I'm 6 years old")
print(encoded_input)
'''
句子对：不是列表的形式

输出的结果如下所示：
{'input_ids': [101, 1731, 1385, 1132, 1128, 136, 102, 146, 112, 182, 127, 1201, 1385, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

'''

print(tokenizer.decode(encoded_input["input_ids"]))

'''
输出的结果如下所示：
[CLS] How old are you? [SEP] I'm 6 years old [SEP]
'''
