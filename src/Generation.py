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

    # def beam_search(self, logits):
    #     from tensor2tensor.utils.beam_search import beam_search
    #     beam_search()
    #     pass
    def sample_top_p(self, logits, top_p=0.9, *args, **kwargs):
        """Chooses most probable logits with cumulative probabilities upto top_p.

        Sets the remaining logits to negative infinity.

        Args:
          logits: Input logits for next token.
          top_p: Float tensor with a value >=0 and < 1.0

        Returns:
          Logits with top_p filtering applied.
        """
        logits = tf.constant(logits[None,:])
        logits_shape = get_shape_list(logits)

        sorted_indices = tf.argsort(logits, direction="DESCENDING")
        # Flatten logits as tf.gather on TPU needs axis to be compile time constant.
        range_for_gather = tf.expand_dims(tf.range(0, logits_shape[0]), axis=1)
        range_for_gather = tf.tile(range_for_gather * logits_shape[1],
                                   [1, logits_shape[1]]) + sorted_indices
        flattened_logits = tf.reshape(logits, [-1])
        flattened_sorted_indices = tf.reshape(range_for_gather, [-1])
        sorted_logits = tf.reshape(
            tf.gather(flattened_logits, flattened_sorted_indices),
            [logits_shape[0], logits_shape[1]])
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative probability above the threshold.
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep the first token above threshold.
        sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = tf.concat([
            tf.zeros_like(sorted_indices_to_remove[:, :1]),
            sorted_indices_to_remove[:, 1:]
        ], -1)

        # Scatter sorted indices to original indexes.
        indices_to_remove = scatter_values_on_batch_indices(sorted_indices_to_remove,
                                                            sorted_indices)
        top_p_logits = set_tensor_by_indices_to_value(logits, indices_to_remove,
                                                      np.NINF)

        topk_ids = tf.random.categorical(
            top_p_logits, dtype=tf.int32, num_samples=1)
        # topk_log_probs = tf.gather(
        #     original_log_probs, topk_ids, axis=1, batch_dims=1)
        # print(sorted_indices.numpy()[0])
        # print(top_p_logits.numpy()[0])
        # top_p_logits = np.nan_to_num(top_p_logits.numpy()[0])
        # print(np.random.choice(sorted_indices.numpy()[0], p=top_p_logits))
        # print(top_p_logits)
        # sampled_logits = tf.cond(
        #     self.top_p < 1,
        #     lambda: sample_top_p(sampled_logits, self.top_p),
        #     lambda: sampled_logits)
        # topk_ids = tf.random.categorical(), dtype=tf.int32, num_samples=1)
        # topk_log_probs = tf.gather(
        #     original_log_probs, topk_ids, axis=1, batch_dims=1)
        return topk_ids.numpy()[0][0]


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

            sample_token = sampling_method(y[0][sample_index], *args, **kwargs)
            tokens_generated.append(sample_token)
            prompt_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        print(f"Generated {num_tokens_generated} tokens")
        txt = " ".join(
            [self.detokenize(_) for _ in prompt_tokens   ]
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

    def on_epoch_end(self, epoch, logs=None, sampling_method=Generator.sample_top_k):
        if (epoch + 1) % self.print_every != 0:
            return
        generator = Generator(self.model, self.seq_len, self.vocab)
        txt = generator.generate(self.prompt_txt, self.max_tokens, sampling_method)
        print(f"\nGenerated:\n{txt}\n")

    @staticmethod
    def create(start_prompt, seq_len, vocab, gen_len=100):
        return




def scatter_values_on_batch_indices(values, batch_indices):
    """Scatter `values` into a tensor using `batch_indices`.

    Args:
      values: tensor of shape [batch_size, vocab_size] containing the values to
        scatter
      batch_indices: tensor of shape [batch_size, vocab_size] containing the
        indices to insert (should be a permutation in range(0, n))

    Returns:
      Tensor of shape [batch_size, vocab_size] with values inserted at
      batch_indices
    """
    tensor_shape = get_shape_list(batch_indices)
    broad_casted_batch_dims = tf.reshape(
        tf.broadcast_to(
            tf.expand_dims(tf.range(tensor_shape[0]), axis=-1), tensor_shape),
        [1, -1])
    pair_indices = tf.transpose(
        tf.concat([broad_casted_batch_dims,
                   tf.reshape(batch_indices, [1, -1])], 0))
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), tensor_shape)


def set_tensor_by_indices_to_value(input_tensor, indices, value):
    """Where indices is True, set the value in input_tensor to value.

    Args:
      input_tensor: float (batch_size, dim)
      indices: bool (batch_size, dim)
      value: float scalar

    Returns:
      output_tensor: same shape as input_tensor.
    """
    value_tensor = tf.zeros_like(input_tensor) + value
    output_tensor = tf.where(indices, value_tensor, input_tensor)
    return output_tensor


def get_shape_list(tensor):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape
