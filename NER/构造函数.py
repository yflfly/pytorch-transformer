# coding:utf-8
def __init__(self, config):
    super().__init__(config)
    self.config = config

    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.pooler = BertPooler(config)

    self.init_weights()
