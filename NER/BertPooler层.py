# pooler层的输入是transformer最后一层的输出，[batch_size, seq_length, hidden_size]
def forward(self, hidden_states):
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.

    # 取出每一句的第一个单词，做全连接和激活。
    # 得到的输出可以用来分类等下游任务（即将每个句子的第一个单词的表示作为整个句子的表示）
    first_token_tensor = hidden_states[:, 0]
    pooled_output = self.dense(first_token_tensor)
    pooled_output = self.activation(pooled_output)
    return pooled_output