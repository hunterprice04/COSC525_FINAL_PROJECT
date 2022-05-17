import os

import tensorflow as tf

from src.data.Embeddings import Embeddings
from src.utils import Logger


class TextDataset(Logger.Wrapper):

    def __init__(self, config, verbosity=1):
        super().__init__(verbosity=verbosity)
        self.print_1(f"# [TextDataset]:\t{config}")
        self.config = config
        self.file_path: str = config.MODEL.DATA_PATH
        self.label: str = os.path.basename(self.file_path).split(".")[0]
        self.window_size: int = config.MODEL.WINDOW_SIZE
        self.stride: int = config.MODEL.STRIDE
        self.dataset_ids = None
        self.dataset_oh = None

    def __str__(self):
        result = f"# Dateset Summary [label=`{self.label}`]:\n"
        result += "=> Configuration:\n"
        result += f"\t{self.config.MODEL}\n"
        show_max = 10
        result += f"=> {show_max} Samples:\n"
        for i, (x, y) in enumerate(self.dataset_ids.take(show_max)):
            if i > show_max:
                break
            result += f'\t{i + 1}. {self.config.EMBED.to_tfstring(x):} => {self.config.EMBED.to_tfstring(y)}\n'
        result += str(self.config.EMBED)
        return result

    @staticmethod
    def __split_xy__(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    def read(self, encoding="utf-8"):
        self.print_2(f"# [TextDataset - read]:")
        self.print_2(f"1. Reading `{self.file_path}`...")
        with open(self.file_path, 'r', encoding=encoding) as f:
            text = f.read()

        text_char_list = list(text)
        self.print_1(f"\t{len(text_char_list)} total chars in text")

        embeddings = Embeddings(text, label=self.label, encoding=encoding, verbosity=self.verbosity)
        self.config.set_embeddings(embeddings)

        self.print_2("2. Converting __items__ to ids...")
        ids = self.config.EMBED.chars_to_ids(text_char_list)
        self.print_2(f"\tids: (len={len(ids)} min={tf.reduce_min(ids)}, max={tf.reduce_max(ids)})")
        data = tf.data.Dataset.from_tensor_slices(ids)
        self.print_2(f"\tdata: {data}")

        self.print_2("3. Creating sequences...")

        def create_seqs(chunk):
            return chunk.batch(self.window_size + 1, drop_remainder=True)

        data = data.window(size=self.window_size + 1, shift=self.stride, drop_remainder=True)
        data = data.flat_map(create_seqs)
        self.print_2(f"\tdata: {data}")

        self.print_2("4. Splitting into inputs and targets...")
        self.dataset_ids = data.map(self.__split_xy__)
        self.print_2(f"\tdataset_ids: {self.dataset_ids}")

        self.print_2("5. One-hot encoding...")
        self.dataset_oh = self.dataset_ids.map(
            lambda x, y: (self.config.EMBED.to_onehot(x), self.config.EMBED.to_onehot(y)))
        self.print_2(f"\tdataset_oh: {self.dataset_oh}")

        return self

    def create_batches(self, batch_size: int, buffer_size: int = 10000):
        dataset_shuf = self.dataset_oh.shuffle(buffer_size)
        dataset = dataset_shuf.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        # dataset = (self.dataset_oh.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        #            .prefetch(tf.data.experimental.AUTOTUNE))
        return dataset_shuf, dataset
