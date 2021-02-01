# coding:utf-8
from transformers import BertConfig

# 在v2.10.0中使用的自动识别的类,但在此次源码分享中仅以Bert模型为例
# from transformers import  AutoConfig,
config = BertConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    id2label=label_map,
    label2id={label: i for i, label in enumerate(labels)},
    cache_dir=model_args.cache_dir,
)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast,
    do_lower_case=args.do_lower_case
)