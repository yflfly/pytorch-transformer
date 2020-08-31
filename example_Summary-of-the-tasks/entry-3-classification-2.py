# coding:utf-8
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")  # 第一次运行时需要进行预训练模型的下载
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

paraphrase_classification_logits = model(**paraphrase)[0]
not_paraphrase_classification_logits = model(**not_paraphrase)[0]

paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

# Should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
# not paraphrase: 10%
# is paraphrase: 90%

# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
# not paraphrase: 94%
# is paraphrase: 6%

'''
第一次运行输出结果如下所示：
Downloading: 100%|██████████| 433/433 [00:00<00:00, 1.41kB/s]
Downloading: 100%|██████████| 213k/213k [00:13<00:00, 15.5kB/s]
Downloading: 100%|██████████| 433M/433M [02:31<00:00, 2.85MB/s]
not paraphrase: 10%
is paraphrase: 90%
not paraphrase: 94%
is paraphrase: 6%

当模型下载完成之后，再运行的结果输出结果如下所示：
not paraphrase: 10%
is paraphrase: 90%
not paraphrase: 94%
is paraphrase: 6%
注：不再进行模型的下载
'''

'''
判断两个句子是否是一篇文章中，可以看成对两个合在一起的句子进行 二分类 问题
'''
