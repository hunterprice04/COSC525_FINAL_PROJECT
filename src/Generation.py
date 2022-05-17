import os
import random
import string

import numpy as np
import tensorflow as tf


class Generator:
    def __init__(self, model, seq_len, vocab):
        self.model = model
        self.seq_len = seq_len
        self.vocab = vocab
        self.word_to_index = {word: index for index, word in enumerate(vocab)}

    def sample_top_k(self, logits, top_k=10):
        logits, indices = tf.math.top_k(logits, k=top_k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.vocab[number]

    def generate(self, start_prompt, max_tokens):
        prompt_tokens = [self.word_to_index.get(_, 1) for _ in start_prompt.lower().split()]
        prompt_tokens = [_ for _ in prompt_tokens]

        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= max_tokens:
            pad_len = self.seq_len - len(prompt_tokens)
            sample_index = len(prompt_tokens) - 1
            if pad_len < 0:
                x = prompt_tokens[:self.seq_len]
                sample_index = self.seq_len - 1
            elif pad_len > 0:
                x = prompt_tokens + [0] * pad_len
            else:
                x = prompt_tokens
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0)
            sample_token = self.sample_top_k(y[0][sample_index])
            tokens_generated.append(sample_token)
            prompt_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        print(f"Generated {num_tokens_generated} tokens")
        txt = " ".join(
            [self.detokenize(_) for _ in prompt_tokens]
        )
        return txt


class GenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self, prompt_txt, max_tokens, seq_len, vocab, top_k=10, print_every=1):
        self.max_tokens = max_tokens
        self.seq_len = seq_len
        self.prompt_txt = prompt_txt
        self.vocab = vocab
        self.print_every = print_every
        self.k = top_k

    def detokenize(self, number):
        return self.vocab[number]

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        generator = Generator(self.model, self.seq_len, self.vocab)
        txt = generator.generate(self.prompt_txt, self.max_tokens)
        print(f"\nGenerated:\n{txt}\n")

    @staticmethod
    def create(start_prompt, seq_len, vocab, gen_len=100):
        return
