import tensorflow as tf
import tensorflow_text as tf_text


class Tokenizer(tf.keras.layers.TextVectorization):
    def __init__(self, dataset, model_config):
        print("[TOKENIZER]: Initializing...")
        max_vocab_size, max_seq_len = model_config.VOCAB_SZ, model_config.MAX_LEN
        super().__init__(standardize=self.preprocess_txt,
                         max_tokens=max_vocab_size - 1,
                         output_mode="int",
                         output_sequence_length=max_seq_len + 1)
        self.adapt(dataset)

    def preprocess_txt(self, input_string):
        # Split accecented characters.
        text = tf_text.normalize_utf8(input_string, 'NFKD')
        text = tf.strings.lower(text)
        # Keep space, a to z, and select punctuation.
        text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
        # Add spaces around punctuation.
        text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
        # Strip whitespace.
        text = tf.strings.strip(text)

        text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
        return text
