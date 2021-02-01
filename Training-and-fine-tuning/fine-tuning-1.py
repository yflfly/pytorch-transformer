# coding:utf-8
from transformers import BertForSequenceClassification
from transformers import AdamW

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
model.train()

optimizer = AdamW(model.parameters(), lr=1e-5)
