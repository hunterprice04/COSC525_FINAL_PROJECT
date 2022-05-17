import os
import random
import string

import numpy as np
import tensorflow as tf
from .top_p import sample_top_p

class Generator:
    def __init__(self, model, seq_len, vocab):
        self.model = model
        self.seq_len = seq_len
        self.vocab = vocab
        self.word_to_index = {word: index for index, word in enumerate(vocab)}
        self.sampling_funcs = {
            'greedy': self.greedy,
            'random': self.sample_random,
            'top_k': self.sample_top_k,
            'top_p': self.sample_top_p
        }

    def greedy(self, logits, *args, **kwargs):
        log_probs = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
        return tf.argmax(log_probs, axis=-1).numpy()

    def sample_random(self, logits, temp=0.5, *args, **kwargs):
        if tf.rank(logits) > 2:
            logits = tf.squeeze(tf.squeeze(logits, axis=0), 0)

        logits = tf.cast(logits, tf.float64)
        conditional_probability = logits / temp
        exp_preds = tf.exp(conditional_probability)
        conditional_probability = exp_preds / tf.reduce_sum(exp_preds)

        probas = np.random.multinomial(1, conditional_probability, 1)
        return np.argmax(probas)

    def sample_top_k(self, logits, top_k=10, *args, **kwargs):
        logits, indices = tf.math.top_k(logits, k=top_k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def beam_search(self, logits, initial_ids=None, beam_size=2, alpha=0.6, *args, **kwargs):
        logits = tf.constant(logits[None,:], dtype=tf.float32)
        initial_ids = tf.constant(np.array(initial_ids)[None,:], dtype=tf.int32)
        print(initial_ids)
        print(len(self.vocab))
        from tensor2tensor.utils.beam_search import beam_search
        pred = beam_search(
            symbols_to_logits_fn=lambda ids: tf.tile(logits, tf.constant([beam_size,1], dtype=tf.float32)),  # A hack to make it work with our architecture
            initial_ids=initial_ids,  # A hack to make it work with our architecture
            beam_size=beam_size,
            decode_length=1,
            vocab_size=len(self.vocab)+1,
            eos_id=0,
            alpha=alpha  # No idea what this does
        )
        print(pred)
        return pred

    def sample_top_p(self, logits, top_p=0.9, *args, **kwargs):
        return sample_top_p(logits, top_p=0.9, *args, **kwargs)


    def detokenize(self, number):
        return self.vocab[number]

    def generate(self, start_prompt, max_tokens, sampling_method, *args, **kwargs):
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
            # sample_token = self.sample_top_k(y[0][sample_index])

            sample_token = sampling_method(y[0][sample_index], *args, **kwargs, initial_ids=prompt_tokens)
            tokens_generated.append(sample_token)
            prompt_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        print(f"Generated {num_tokens_generated} tokens")
        txt = " ".join(
            [self.detokenize(_) for _ in prompt_tokens]
        )
        return txt


class GenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self, prompt_txt, max_tokens, seq_len, vocab, top_k=10, print_every=1, tb_file_writer=None):
        self.max_tokens = max_tokens
        self.seq_len = seq_len
        self.prompt_txt = prompt_txt
        self.vocab = vocab
        self.print_every = print_every
        self.k = top_k
        self.tb_file_writer = tb_file_writer

    def detokenize(self, number):
        return self.vocab[number]

    def on_epoch_end(self, epoch, logs=None, sampling_method=Generator.sample_top_k):
        if (epoch + 1) % self.print_every != 0:
            return
        generator = Generator(self.model, self.seq_len, self.vocab)
        for name, f in generator.sampling_funcs.items():
            txt = generator.generate(self.prompt_txt, self.max_tokens, sampling_method)
            print(f"\n{name} generated:\n{txt}\n")
            if self.tb_file_writer is not None:
                self.tb_file_writer.text(name, txt, epoch)

    @staticmethod
    def create(start_prompt, seq_len, vocab, gen_len=100):
        return

