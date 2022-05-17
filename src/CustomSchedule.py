import tensorflow as tf
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule


class CustomSchedule(LearningRateSchedule):
    def __init__(self, emb_dim, warmup_steps=4000, **kwargs):
        super().__init__()
        self.emb_dim = emb_dim
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        ed = tf.cast(self.emb_dim, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(ed) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {'emb_dim': self.emb_dim, 'warmup_steps': self.warmup_steps}
