# coding:utf-8
from transformers import pipeline

nlp = pipeline("sentiment-analysis")
result = nlp('I hate you')[0]
print(f"label:{result['label']},with score:{round(result['score'], 4)}")

result = nlp("I love you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

result = nlp('I hate you')
print(result)
'''
输出结果如下所示：
label:NEGATIVE,with score:0.9991

label: POSITIVE, with score: 0.9999

[{'label': 'NEGATIVE', 'score': 0.9991129040718079}]
'''