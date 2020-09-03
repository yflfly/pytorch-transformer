# coding:utf-8
from transformers import pipeline

text_generator = pipeline("text-generation")

print(text_generator("As far as I am concerd, I will", max_length=50, do_sample=False))

'''
Here, the model generates a random text with a total maximal length of 50 tokens from context “As far as I am concerned, I will”. The default arguments of PreTrainedModel.generate() can be directly overriden in the pipeline, as is shown above for the argument max_length.

第一次运行同样需要下载相关的模型
Downloading: 100%|██████████| 665/665 [00:00<00:00, 167kB/s]
Downloading: 100%|██████████| 1.04M/1.04M [00:01<00:00, 603kB/s]
Downloading: 100%|██████████| 456k/456k [00:17<00:00, 26.1kB/s]
Downloading: 100%|██████████| 230/230 [00:00<00:00, 57.7kB/s]
D:\ccidit\gjcs_autocase\venv\lib\site-packages\transformers\modeling_auto.py:798: FutureWarning: The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use `AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and `AutoModelForSeq2SeqLM` for encoder-decoder models.
  FutureWarning,
Downloading: 100%|██████████| 548M/548M [25:38<00:00, 356kB/s] 
Some weights of GPT2LMHeadModel were not initialized from the model checkpoint at gpt2 and are newly initialized: ['h.0.attn.masked_bias', 'h.1.attn.masked_bias', 'h.2.attn.masked_bias', 'h.3.attn.masked_bias', 'h.4.attn.masked_bias', 'h.5.attn.masked_bias', 'h.6.attn.masked_bias', 'h.7.attn.masked_bias', 'h.8.attn.masked_bias', 'h.9.attn.masked_bias', 'h.10.attn.masked_bias', 'h.11.attn.masked_bias', 'lm_head.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Setting `pad_token_id` to 50256 (first `eos_token_id`) to generate sequence
输出结果如下所示：
[{'generated_text': 'As far as I am concerd, I will not be able to tell you how many times I have been told that I am not allowed to speak to the press. I have been told that I am not allowed to speak to the press. I'}]

'''
