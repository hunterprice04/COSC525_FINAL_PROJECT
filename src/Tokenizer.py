import string

import tensorflow as tf


class Tokenizer(tf.keras.layers.TextVectorization):
    def __init__(self, dataset, max_vocab_size, max_seq_len):
        super().__init__(standardize=self.preprocess_txt,
                         max_tokens=max_vocab_size - 1,
                         output_mode="int",
                         output_sequence_length=max_seq_len + 1)
        self.adapt(dataset)
        self.vocab = self.get_vocabulary()

    def preprocess_txt(self, input_string):
        # Preprocessing for word-level model
        s1 = tf.strings.lower(input_string)
        return tf.strings.regex_replace(s1, f"([{string.punctuation}])", r" \1")
