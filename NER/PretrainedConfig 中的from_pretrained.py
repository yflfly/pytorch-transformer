# coding:utf-8
def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    r"""
    Args:
        pretrained_model_name_or_path (:obj:`string`)    名称          
        cache_dir (:obj:`string`, `optional`)             缓存文件夹  
        kwargs (:obj:`Dict[str, any]`, `optional`)        其他参数用 `Dict[str, any]`的形式     
        force_download (:obj:`bool`, `optional`, defaults to :obj:`False`)  覆盖文件夹中之前下载文件             
        resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`) 是否继续下载
        proxies (:obj:`Dict`, `optional`)  代理服务器地址
    Returns:
        :class:`PretrainedConfig`: An instance of a configuration object

    Examples::
        config 文件可以采取三种方式，bert名称、bert文件夹地址、config文件地址
        config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
        config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
        config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
        config = BertConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
        assert config.output_attention == True
        config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                           foo=False, return_unused_kwargs=True)
        assert config.output_attention == True
        assert unused_kwargs == {'foo': False}
    """
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    return cls.from_dict(config_dict, **kwargs)
