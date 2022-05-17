import random
import tensorflow as tf


class Dataset(tf.data.TFRecordDataset):
    def __init__(self, file_pth, shuffle=True):
        print(f"[DATASET]: Loading {file_pth}")
        if shuffle:
            random.shuffle(file_pth)
        # Shuffle the data and create batches
        super().__init__(file_pth)

    def create_batch_sequences(self, tokenizer, batch_sz, buf_sz=1000, shuffle=True):
        def __create_sequences(txt):
            txt = tf.expand_dims(txt, -1)
            txt_tok = tokenizer(txt)
            return txt_tok[:, :-1], txt_tok[:, 1:]

        data = self
        if shuffle:
            data = data.shuffle(buffer_size=buf_sz)
        batched = data.batch(batch_sz)
        sequences = batched.map(__create_sequences).prefetch(tf.data.AUTOTUNE)
        return sequences
