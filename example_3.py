# coding:utf-8
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# sequence_a = "HuggingFace is based in NYC"
# sequence_b = "Where is HuggingFace based?"
# encoded_dict = tokenizer(sequence_a, sequence_b)
# decoded = tokenizer.decode(encoded_dict["input_ids"])

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sequence_a = "HuggingFace is based in NYC"
sequence_b = "Where is HuggingFace based?"
encoded_dict = tokenizer(sequence_a, sequence_b)
decoded = tokenizer.decode(encoded_dict["input_ids"])
