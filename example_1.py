# coding:utf-8
from transformers import pipeline

def h1():
    classifier = pipeline('sentiment-analysis')
    print(classifier('We are very happy to show you the 🤗 Transformers library.'))
    print(classifier('很开心看到你！'))

    '''
    利用transformer封装好的NLP相关的方法进行NLP任务的处理
    上述实例进行NLP任务中  情感分析任务
    
    运行的结果如下所示：
    Downloading: 100%|██████████| 230/230 [00:00<00:00, 57.2kB/s]
    [{'label': 'POSITIVE', 'score': 0.9997795224189758}]
    [{'label': 'POSITIVE', 'score': 0.8704843521118164}]
    '''

    classifier = pipeline('sentiment-analysis')
    results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])

    for result in results:
        print(f"label:{result['label']},with score:{round(result['score'], 4)}")

    ''''
    输出的结果为：
    label:POSITIVE,with score:0.9998
    label:NEGATIVE,with score:0.5309
    '''


from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
classifier = pipeline('sentiment-analysis', model=model)
results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])

for result in results:
    print(f"label:{result['label']},with score:{round(result['score'], 4)}")

'''
指定了模型
输出的结果为：

'''