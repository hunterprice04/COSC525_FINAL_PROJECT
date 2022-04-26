import tensorflow as tf

from src.data import Vocab


class Embeddings(Vocab):

    def __init__(self, text, label=None, encoding="utf-8", verbosity=1):
        super().__init__(text=text, label=label, encoding=encoding, verbosity=verbosity)
        self.print_1(f"# [Embeddings - init]:\tlabel={label}\tencoding={encoding}")

    def to_tfstring(self, x):
        x_ids = self.to_ids(x)
        x_chars = self.ids_to_chars(x_ids)
        return tf.strings.reduce_join(x_chars, axis=-1)

    def to_string(self, x):
        tf_str = self.to_tfstring(x)
        return tf_str.numpy()[0].decode(self.encoding)

    def to_onehot(self, x):
        x_ids = self.to_ids(x)
        # return tf.expand_dims(tf.one_hot(x_ids, len(self), dtype=tf.int64), 0)
        return tf.one_hot(x_ids, self.vocab_size, dtype=tf.int64)
        # return to_categorical(x_ids, num_classes=len(self))

    def to_ids(self, x):
        if isinstance(x, str):
            self.print_2("* to_ids -> input is a String; converting to IDs first...")
            x = self.chars_to_ids(list(x))
        elif x.shape[-1] == len(self):
            self.print_2("* to_ids -> input is One-Hot Encoded; converting to IDs first...")
            x = tf.argmax(x, axis=-1)
        return x

    def get_vocab(self):
        return self.get()
