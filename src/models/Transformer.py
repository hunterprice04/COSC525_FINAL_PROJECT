import tensorflow as tf


class Transformer:

    def __init__(self, training=True):
        self.training = training
        self.encoder = None
        self.decoder = None


class Encoder(tf.keras.layers.Layer):

    def __init__(self, hidden_size, num_heads, filter_size, dropout_rate, training=True):
        super(Encoder, self).__init__()
        print("Encoder")
        self._training = training
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        # 1. Attention Layer (MultiHeadedAttention)
        # 2. Layer Normalization
        # 3. Dropout

    def call(self, x, training=True):
        # TODO
        return self.encoder


class Decoder(tf.keras.layers.Layer):

    def __init__(self, hidden_size, num_heads, filter_size, dropout_rate, training=True):
        super(Decoder, self).__init__()
        print("Encoder")
        self._training = training
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

    def call(self, x, training=True):
        # TODO
        return self.encoder
