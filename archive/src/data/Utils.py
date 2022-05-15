import numpy as np
import tensorflow as tf


class Utils:

    @staticmethod
    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attentions logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        """
        :param size:
        :return:
        """

        """
        mask = 1 - tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
        mask = mask[tf.newaxis, tf.newaxis, :, :]
        """
        return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # (seq_len, seq_len)

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @staticmethod
    def get_positional_encoding(seq_len, hidden_size, reverse=False):
        """Creates a tensor that encodes positional information.
        Args:
          seq_len: int scalar tensor, sequence length.
          hidden_size: int scalar, the hidden size of continuous representation.
          reverse: bool, whether to reverse the sequence. Defaults to False.
        Returns:
          positional_encoding: float tensor of shape [seq_len, hidden_size], the
            tensor that encodes positional information.
        """
        distances = tf.cast(tf.range(seq_len), 'float32')
        hidden_size //= 2
        inverse_frequencies = 1 / (
                10000 ** (tf.cast(tf.range(hidden_size), 'float32') / (hidden_size - 1)))
        positional_encoding = tf.einsum('i,j->ij', distances, inverse_frequencies)
        positional_encoding = tf.concat([tf.sin(positional_encoding),
                                         tf.cos(positional_encoding)], axis=1)
        return positional_encoding

    @staticmethod
    def get_positional_encoding_old(position, d_model):
        angle_rads = Utils.get_angles(np.arange(position)[:, np.newaxis],
                                      np.arange(d_model)[np.newaxis, :],
                                      d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)
