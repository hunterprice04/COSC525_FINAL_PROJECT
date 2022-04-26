import os


class Utils:

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
