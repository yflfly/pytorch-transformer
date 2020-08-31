# coding:utf-8
from transformers import pipeline

nlp = pipeline("question-answering")

context = r'''Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.'''

result = nlp(question="What is extractive question answering?", context=context)
print(result)

print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

result = nlp(question="What is a good example of a question answering dataset?", context=context)
print(result)

'''
输出结果如下所示：
{'score': 0.6185734643682169, 'start': 33, 'end': 95, 'answer': 'the task of extracting an answer from a text given a question.'}
Answer: 'the task of extracting an answer from a text given a question.', score: 0.6186, start: 33, end: 95
{'score': 0.5039562316275037, 'start': 146, 'end': 160, 'answer': 'SQuAD dataset,'}

'''