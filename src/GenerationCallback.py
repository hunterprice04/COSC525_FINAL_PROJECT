import os
import random
import string

import numpy as np
import tensorflow as tf


class GenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_tokens, seq_len, start_tokens, index_to_word, top_k=10, print_every=1):
        self.max_tokens = max_tokens
        self.seq_len = seq_len
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = list(self.start_tokens)
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.seq_len - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.seq_len]
                sample_index = self.seq_len - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join([self.detokenize(_) for _ in self.start_tokens + tokens_generated])
        print(f"Generated:\n{txt}\n")

    @staticmethod
    def create(start_prompt, seq_len, vocabulary, gen_len=100):
        # Tokenize starting prompt
        word_to_index = {word: index for index, word in enumerate(vocabulary)}
        prompt_tokens = [word_to_index.get(_, 1) for _ in start_prompt.lower().split()]
        return GenerationCallback(gen_len, seq_len, prompt_tokens, vocabulary)
