import os

from transformers import AutoTokenizer, TFAutoModelForCausalLM

from src.utils.Config import Config
from src.utils.GPUtils import GPUtils
import tensorflow as tf

class ModelUtils:

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
    def generate(model, tokenizer, text, max_length=512, temperature=0.7, top_k=50, top_p=0.9, print_mem=True):
        model, tokenizer = ModelUtils.get_model_and_tokanizer(model, tokenizer, print_mem)
        if print_mem:
            GPUtils.print_usage()
        # Generate a sequence of tokens
        input_ids = tokenizer.encode(text, return_tensors='pt')

    @staticmethod
    def get_model_and_tokanizer(model_name, tokenizer_name=None, print_mem=True):
        if tokenizer_name is None:
            print(f"W: Tokanizer wasn't specified explicitly, using same as model: {model_name}")
            tokenizer_name = model_name

        if print_mem:
            GPUtils.print_usage()

        # Initialize tokenizer
        tokenizer_name = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        # Download model and configuration from huggingface.co and cache.
        model_name = TFAutoModelForCausalLM.from_pretrained(
            model_name
        )
        if print_mem:
            GPUtils.print_usage()

        return model_name, tokenizer_name

    @staticmethod
    def print_outputs(outputs, strip=True):
        print("=" * 80)
        print(f"# OUTPUTS: len={len(outputs)}")
        for i in range(len(outputs)):
            oup = outputs[i]
            if strip:
                oup = oup.split("\n")
                oup = [x for x in oup if x != '']
                oup = "\n".join(oup)
                # oup = oup.strip()
            print("-" * 80)
            print(f"# OUTPUT[{i}]: len={len(oup)}")
            print(oup)
