import tensorflow as tf

from src import Logger


class Vocab(Logger.Wrapper):

    def __init__(self, text, label=None, encoding='utf-8', verbosity=1):
        super().__init__(verbosity=verbosity)
        self.print_2(f"# [Vocab]:\tlabel={label}\tencoding={encoding}\tverbosity={verbosity}")
        self.label = label
        self.encoding = encoding
        self.__items__ = sorted(list(set(text)))
        self.chars_to_ids, self.ids_to_chars = self.create_enc_dec(self.__items__, self.encoding)
        self.vocab_size = self.chars_to_ids.vocabulary_size()

    def create_enc_dec(self, vocab, encoding):
        chars_to_ids = tf.keras.layers.StringLookup(vocabulary=vocab, encoding=encoding,
                                                    mask_token=None)
        ids_to_chars = tf.keras.layers.StringLookup(vocabulary=vocab, invert=True, encoding=encoding,
                                                    mask_token=None)
        return chars_to_ids, ids_to_chars

    def __str__(self):
        alpha = [c for c in self.__items__ if c.isalpha()]
        num = [c for c in self.__items__ if c.isnumeric()]
        symbols = [c for c in self.__items__ if not c.isalnum()]
        return f"# Vocab Summary [label=`{self.label}`]:\n" \
               f"\t* {len(self)} unique chars\n" \
               f"\t* {len(alpha)} alpha\t{len(num)} numeric\t{len(symbols)} symbols\n"

    def get(self):
        return self.chars_to_ids.get_vocabulary()

    # Emulate the needed methods for self.__items__
    def __len__(self):
        return self.vocab_size

    def __getitem__(self, key):
        return self.__items__[key]

    def __contains__(self, item):
        return item in self.__items__

    def __iter__(self):
        return iter(self.__items__)

    def __size__(self):
        return len(self.__items__)
