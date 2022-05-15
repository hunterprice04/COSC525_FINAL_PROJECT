import tensorflow as tf
from tensorflow.python.framework import tensor_shape


class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, proj_weights=None, kernel_initializer=None):
        super(OutputLayer, self).__init__()
        self.proj_weights = proj_weights
        self.output_dim = output_dim
        self.layer_weights = None
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        if self.proj_weights is None:
            input_dim = tensor_shape.dimension_value(input_shape[-1])
            self.layer_weights = self.add_weight(
                'output_layer_weights',
                shape=[input_dim, self.output_dim],
                initializer=self.kernel_initializer,
                trainable=True)
        super(OutputLayer, self).build(input_shape)

    def call(self, x):
        batch, sequence, d_model = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[-1]
        h_flat = tf.reshape(x, [-1, d_model])

        if self.proj_weights is None:
            out = tf.matmul(h_flat, self.layer_weights)
        else:
            out = tf.matmul(h_flat, self.porj_weights, transpose_b=True)
        out = tf.reshape(out, [batch, sequence, self.output_dim])
        return out
