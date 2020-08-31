# coding:utf-8
from transformers import pipeline
classifier = pipeline('sentiment-analysis')

'''
先进行模型的下载：
Downloading: 100%|#####################################################| 411/411 [00:00<00:00, 164kB/s]
Downloading: 100%|##################################################| 213k/213k [00:07<00:00, 28.1kB/s]
Downloading: 100%|#####################################################| 629/629 [00:00<00:00, 281kB/s]
Downloading: 100%|####################################################| 230/230 [00:00<00:00, 80.5kB/s]
Downloading: 100%|################################################| 268M/268M [2:03:00<00:00, 36.3kB/s]
'''

print(classifier('we are happy'))

'''
输出的结果如下所示：
[{'label': 'NEGATIVE', 'score': 0.98544323}]
'''

'''
上面的代码来源网址：
https://huggingface.co/transformers/quicktour.html
'''