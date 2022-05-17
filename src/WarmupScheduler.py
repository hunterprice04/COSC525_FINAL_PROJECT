import tensorflow as tf


class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, emb_dim, warmup_steps):
        super().__init__()
        self.emb_dim = tf.cast(emb_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.emb_dim) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        c = super().get_config()
        c.update({
            "emb_dim": self.emb_dim,
            "warmup_steps": self.warmup_steps,
        })
        return c
