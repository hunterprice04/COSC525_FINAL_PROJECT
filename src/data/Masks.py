import tensorflow as tf


class Masks:
    @staticmethod
    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # (seq_len, seq_len)
