import os

from src import Config


class Utils:

    @staticmethod
    def create_model_name(config: Config, model_type: str):
        D, TR = config.DATA, config.TRAINING
        n_l = config.EMBED.label
        n_d = f"WS-{D.WINDOW_SIZE}_ST-{D.STRIDE}"
        n_hs = [str(hs) for hs in TR.HIDDEN_STATE_SIZE]
        n_t = f"BS-{TR.BATCH_SIZE}_HS-{'-'.join(n_hs)}_DR-{TR.DROPOUT}_LR-{TR.LR}"
        return f'model_{model_type}_{n_l}_{n_d}_{n_t}'

    @staticmethod
    def create_callbacks(base_dir, model, defaults: list = None):
        import tensorflow as tf
        print(f'base_dir: {base_dir}')
        dir_models = os.path.join(base_dir, model.name)
        path_csv = os.path.join(dir_models, 'history.csv')
        print("History CSV:", path_csv)
        path_ckp = os.path.join(dir_models, 'checkpoints.h5')
        print("Checkpoint:", path_ckp)
        path_tb = os.path.join(dir_models, "logs")
        tb_file_writer = tf.summary.create_file_writer(path_tb)
        callbacks = [] if defaults is None else defaults
        callbacks.append(tf.keras.callbacks.CSVLogger(path_csv, separator=",", append=True))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(path_ckp,
                                                            monitor='loss',
                                                            save_best_only=True,
                                                            mode='auto',
                                                            verbose=0))
        os.makedirs(dir_models, exist_ok=True)
        return callbacks, tb_file_writer

    @staticmethod
    def key_val(d):
        return '_'.join(list(map(lambda x: f'{str(x[0])}-{str(x[1])}', d.__items__())))

    @staticmethod
    def tensorflow_shutup():
        """
        Make Tensorflow less verbose
        This function is taken from:
        https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
        Thank you, @Adam Wallner for this awesome solution :)
        """
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

            # noinspection PyPackageRequirements
            import tensorflow as tf
            from tensorflow.python.util import deprecation

            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

            # Monkey patching deprecation src to shut it up! Maybe good idea to disable this once after upgrade
            # noinspection PyUnusedLocal
            def deprecated(
                    date, instructions, warn_once=True
            ):  # pylint: disable=unused-argument
                def deprecated_wrapper(func):
                    return func

                return deprecated_wrapper

            deprecation.deprecated = deprecated

        except ImportError:
            pass

    @staticmethod
    def in_notebook():
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
                return False
        except (ImportError, AttributeError):
            return False
        return True
