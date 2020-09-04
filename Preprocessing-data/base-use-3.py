# coding:utf-8
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

batch_sentences = ["Hello I'm a single sentence",
                   "And another sentence",
                   "And the very very last one"]
encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_inputs)
'''
如果输入的是多个句子，则将所有的句子以列表的形式输入到tokenizer,当句子的长度不一致时，需要进行padding操作
输出结果的如下所示：
{'input_ids': tensor([[ 101, 8667,  146,  112,  182,  170, 1423, 5650,  102],
        [ 101, 1262, 1330, 5650,  102,    0,    0,    0,    0],
        [ 101, 1262, 1103, 1304, 1304, 1314, 1141,  102,    0]]),
         'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0]])}
返回的是一个张量
'''
