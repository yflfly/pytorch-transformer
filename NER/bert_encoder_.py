# BertEncoder层建立了整个transformer构架
# Transformer构架参考：https://zhuanlan.zhihu.com/p/39034683        （BE CAUTIOUS!）
# 现在我假设大家都知道了这个架构，我这里沿袭了上面知乎中某些专有名词的称呼

# ........................................................................
# Transformer中包含若干层(论文中base为12层，large为24层)encoder,每层encoder在代码中就是一个BertLayer。
# 所以下面的代码首先声明了一层layer,然后构造了num_hidden_layers(12 or 24)层相同的layer放在一个列表中，既是self.layer
layer = BertLayer(config)
self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])


# ........................................................................

# ........................................................................
# 下面看其forward函数
def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
    # 看其输入：
    # hidden_states:根据上面所讲，hidden_states就是embedding_output，
    # 其维度为[batch_size, seq_lenght, word_dimension],embedding出来后，多了一个dimension
    # attention_mask:维度[batch_size, 1, 1, seq_length]
    # (to be completed)
    # output_all_encoder_layers:此函数的输出模式，下面会详细讲解

    # 这个函数到底做了什么了？其实很简单，就是做了一个循环，将每一个encoder的输出作为输入输给下一层的encoder，直到12（or24）层循环完毕
    all_encoder_layers = []
    # 遍历所有的encoder,总共有12层或者24层
    for layer_module in self.layer:
        # 每一层的输出hidden_states也是下一层layer_moudle（BertLayer）的输入，这样就连接起来了各层encoder。第一层的输入是embedding_output
        hidden_states = layer_module(hidden_states, attention_mask)
        # 如果output_all_encoded_layers == True:则将每一层的结果添加到all_encoder_layers中
        if output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
    # 如果output_all_encoded_layers == False, 则只将最后一层的输出加到all_encoded_layers中
    if not output_all_encoded_layers:
        all_encoder_layers.append(hidden_states)
    return all_encoder_layers
# 所以output_all_encoded_layers是用来控制输出模式的。
# 这样整个transformer的框架就出来了，下面将讲述框架中的每一层encoder（即BertLayer）是怎么构造的
# ........................................................................
