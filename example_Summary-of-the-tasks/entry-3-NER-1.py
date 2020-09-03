# coding:utf-8
from transformers import pipeline

nlp = pipeline("ner")
sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge which is visible from the window."

print(nlp(sequence))

'''
第一次运行需要下载对应的模型
Downloading: 100%|██████████| 998/998 [00:00<00:00, 500kB/s]
Downloading: 100%|██████████| 213k/213k [00:00<00:00, 244kB/s]  
Downloading: 100%|██████████| 60.0/60.0 [00:00<00:00, 15.0kB/s]
Downloading: 100%|██████████| 230/230 [00:00<00:00, 46.1kB/s]
Downloading: 100%|██████████| 1.33G/1.33G [06:57<00:00, 3.19MB/s]

输出结果如下所示：
[{'word': 'Hu', 'score': 0.999578595161438, 'entity': 'I-ORG', 'index': 1}, {'word': '##gging', 'score': 0.9909763932228088, 'entity': 'I-ORG', 'index': 2}, {'word': 'Face', 'score': 0.9982224702835083, 'entity': 'I-ORG', 'index': 3}, {'word': 'Inc', 'score': 0.9994880557060242, 'entity': 'I-ORG', 'index': 4}, {'word': 'New', 'score': 0.9994344711303711, 'entity': 'I-LOC', 'index': 11}, {'word': 'York', 'score': 0.9993196129798889, 'entity': 'I-LOC', 'index': 12}, {'word': 'City', 'score': 0.9993793964385986, 'entity': 'I-LOC', 'index': 13}, {'word': 'D', 'score': 0.9862582683563232, 'entity': 'I-LOC', 'index': 19}, {'word': '##UM', 'score': 0.9514269232749939, 'entity': 'I-LOC', 'index': 20}, {'word': '##BO', 'score': 0.9336589574813843, 'entity': 'I-LOC', 'index': 21}, {'word': 'Manhattan', 'score': 0.9761654138565063, 'entity': 'I-LOC', 'index': 28}, {'word': 'Bridge', 'score': 0.9914628863334656, 'entity': 'I-LOC', 'index': 29}]

'''
