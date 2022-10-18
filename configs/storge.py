class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used indication to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"


config = {
    "hidden_size": 768,  # default is 768
    "num_attention_heads": 12,  # default is 12
    "attention_probs_dropout_prob": 0.1,
    "position_embedding_type": "absolute",
    "max_position_embeddings": 512,
    "layer_norm_eps": 1e-12,
    "hidden_dropout_prob": 0.1,
    "intermediate_size": 3072,  # default is 3072
    "hidden_act": "gelu",
    "num_hidden_layers": 3,
    "add_cross_attention": True,
}
config = Storage(config)
print(config.hidden_size)
config.hidden_size = 100
print(config.hidden_size)

