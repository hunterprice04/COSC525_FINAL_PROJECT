"""
This file contains the top_p function that was taken from a public implementation at:
https://github.com/tensorflow/models/blob/d55a62919f070ef1d9b4bad8a118c34605ba4427/official/nlp/modeling/ops/sampling_module.py

We have modified it to work with our model.
"""

import numpy as np
import tensorflow as tf


def sample_top_p(logits, top_p=0.9, *args, **kwargs):
    # https://github.com/tensorflow/models/blob/d55a62919f070ef1d9b4bad8a118c34605ba4427/official/nlp/modeling/ops/sampling_module.py
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

    topk_ids = tf.random.categorical(top_p_logits, dtype=tf.int32, num_samples=1)
    return topk_ids.numpy()[0][0]


def scatter_values_on_batch_indices(values, batch_indices):
    # https://github.com/tensorflow/models/blob/d55a62919f070ef1d9b4bad8a118c34605ba4427/official/nlp/modeling/ops/sampling_module.py

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
    # https://github.com/tensorflow/models/blob/d55a62919f070ef1d9b4bad8a118c34605ba4427/official/nlp/modeling/ops/sampling_module.py

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
    # https://github.com/tensorflow/models/blob/d55a62919f070ef1d9b4bad8a118c34605ba4427/official/nlp/modeling/ops/sampling_module.py

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