import tensorflow as tf
import tensorflow_text as tf_text


class Tokenizer(tf.keras.layers.TextVectorization):
    def __init__(self, dataset, model_config):
        print("[TOKENIZER]: Initializing...")
        max_vocab_size, max_seq_len = model_config.VOCAB_SZ, model_config.MAX_LEN
        super().__init__(standardize=self.preprocess_sequence,
                         max_tokens=max_vocab_size - 1,
                         output_mode="int",
                         output_sequence_length=max_seq_len + 1)
        self.adapt(dataset)

    @staticmethod
    def preprocess_sequence(input_string, start=True, end=True):
        # Split accecented characters.
        text = tf_text.normalize_utf8(input_string, 'NFKD')
        text = tf.strings.lower(text)
        # Strip whitespace.
        text = tf.strings.strip(text)
        return Tokenizer.add_start_end_token(text, start, end)

    @staticmethod
    def add_start_end_token(input_string, start=True, end=True):
        text_list = []
        if start:
            text_list.append('[START]')
        text_list.append(input_string)
        if end:
            text_list.append('[END]')
        return tf.strings.join(text_list, separator=' ')
