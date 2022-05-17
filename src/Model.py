import os
import random
import string

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Transformer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_att_heads, state_dims, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_att_heads = num_att_heads
        self.state_dims = state_dims
        self.dropout_rate = dropout_rate
        self.attention = tf.keras.layers.MultiHeadAttention(num_att_heads, embedding_dim)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(state_dims, activation="relu"),
            tf.keras.layers.Dense(embedding_dim)
        ])
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
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_att_heads": self.num_att_heads,
            "state_dims": self.state_dims,
            "dropout_rate": self.dropout_rate,
        })
        return config


class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_dim, warmup_steps):
        super(WarmupScheduler, self).__init__()
        self.emb_dim = tf.cast(embedding_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.emb_dim) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "emb_dim": self.emb_dim,
            "warmup_steps": self.warmup_steps,
        })
        return config


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
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
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0)
    return tf.tile(mask, mult)
