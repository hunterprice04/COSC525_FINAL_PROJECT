import tensorflow as tf


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        pos = tf.range(start=0, limit=max_len, delta=1)
        pos = self.pos_emb(pos)
        x = self.token_emb(x)
        return x + pos

    def get_config(self):
        c = super().get_config()
        c.update({
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return c
