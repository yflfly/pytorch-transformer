# coding:utf-8
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(inputs)

