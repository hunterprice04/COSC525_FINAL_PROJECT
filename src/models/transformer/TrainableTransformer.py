import time

import tensorflow as tf


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Inputs"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Targets")
]


class TrainableTransformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads,
                 dropout_rate, dff, max_seq_len, vocab_size,
                 optimizer="adam", learning_rate=1e-3, rev_embedding_projection=True,
                 grad_clip=False, clip_value=1.0):
        self.rev_embedding_projection = rev_embedding_projection
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.optimizer_t = optimizer
        self.mirrored_strategy = None
        self.grad_clip = grad_clip
        self.clip_value = clip_value

        self.embedding = EmbeddingLayer(
            self.vocab_size, self.d_model)

        self.pos_embedding = PositionEmbeddingLayer(
            self.max_seq_len, self.d_model)

        self.decoder_layers = [DecoderLayer(self.d_model, self.num_heads, self.dff)
                               for _ in range(self.num_layers)]
        self.layer_norm = LayerNormalization(self.d_model)

        if not self.rev_embedding_projection:
            self.output_layer = OutputLayer(self.vocab_size)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        self.accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy(
            name='accuracy')

        self.train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)]

    def call(self, x, training=True, past=None):
        print("Unimplemented!")
        return None

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(self, real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def train(self):
        for epoch in range(EPOCHS):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(train_batches):
                train_step(inp, tar)

                if batch % 50 == 0:
                    print(
                        f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp],
                                         training=True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    def checkpoint_manager(self):
        checkpoint_path = './checkpoints/train'
        ckpt = tf.train.Checkpoint(transformer=transformer,
                                   optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
