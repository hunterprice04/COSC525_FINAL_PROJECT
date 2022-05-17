import tensorflow as tf


class SimpleTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim_emb, att_heads, dim_ffn, num_layers, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.dim_emb = dim_emb
        self.att_heads = att_heads
        self.dim_ffn = dim_ffn
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.attention = tf.keras.layers.MultiHeadAttention(att_heads, dim_emb)
        self.layers = [tf.keras.layers.Dense(dim_ffn, activation="relu") for _ in range(num_layers)]
        self.layers.append(tf.keras.layers.Dense(dim_emb))
        self.feed_forward = tf.keras.Sequential(self.layers)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        inp_shape = tf.shape(inputs)
        batch_sz, seq_len = inp_shape[0], inp_shape[1]
        causal_mask = causal_attention_mask(batch_sz, seq_len, seq_len, tf.bool)
        attention_out = self.attention(inputs, inputs, attention_mask=causal_mask)
        attention_out = self.drop1(attention_out)
        out1 = self.norm1(inputs + attention_out)
        feed_forward_out = self.feed_forward(out1)
        feed_forward_out = self.drop2(feed_forward_out)
        return self.norm2(out1 + feed_forward_out)

    def get_config(self):
        c = super().get_config()
        c.update({
            "dim_emb": self.dim_emb,
            "att_heads": self.att_heads,
            "dim_ffn": self.dim_ffn,
            "dropout_rate": self.dropout_rate,
        })
        return c


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0)
    return tf.tile(mask, mult)
