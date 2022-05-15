import numpy as np
import tensorflow as tf


class Sampling:
    """
    https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc
    """

    @staticmethod
    def random_sampling(logits, temp):
        if tf.rank(logits) > 2:
            logits = tf.squeeze(tf.squeeze(logits, axis=0), 0)

        logits = tf.cast(logits, tf.float64)
        conditional_probability = logits / temp
        exp_preds = tf.exp(conditional_probability)
        conditional_probability = exp_preds / tf.reduce_sum(exp_preds)

        probas = np.random.multinomial(1, conditional_probability, 1)
        return tf.expand_dims(probas, 0)

    @staticmethod
    def random_sampling_bugged(logits, temp):
        if tf.rank(logits) > 2:
            logits = tf.squeeze(logits, axis=0)
        logits = tf.cast(logits, tf.float64)
        if temp is not None:
            logits /= temp

        logits = tf.nn.softmax(logits, axis=1)
        logits = tf.math.log(logits)
        return tf.random.categorical(logits=logits, num_samples=1)

    @staticmethod
    def beam_search(logits):
        # TODO: Implement this
        print("# Unimplemented: beam_search")

    @staticmethod
    def greedy_search(logits, *args, **kwargs):
        return np.argmax(logits, axis=-1)
