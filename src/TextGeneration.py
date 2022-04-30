import json
import os
import pickle
from math import inf

import tensorflow as tf
from pick import pick
from tqdm import tqdm

from src.Sampling import Sampling
from src.data.TextDataset import TextDataset
from src.utils import Logger
from src.utils.Config import TrainingConfig
from src.ModelUtilspy import ModelUtils


class TextGeneration(Logger.Wrapper):

    def __init__(self, config, model_func, verbosity=1):
        super().__init__(verbosity)
        self.config = config
        if model_func is None:
            raise ValueError("# [TextGeneration]: model_func argument cannot be None.")
        self.print_1(f"# [TextGeneration]:\tmodel_func={model_func.__name__}\tverbosity={verbosity}")

        self.model_func = model_func
        self.model_train = self.model_func(config, training=True)
        self.model_pred = self.model_func(config, training=False)
        self.history = None
        if verbosity > 1:
            print(self.model_train.summary())

    def predict(self, prompt, sampling_method, pred_len: int = 50, temp=0.75, include_prompt=False,
                output_as="string", seed=None, verbosity=None):
        if self.model_pred is None:
            raise ValueError("TextGeneration - You must train your model first!")
        if verbosity is None:
            verbosity = self.verbosity
        if seed is not None:
            tf.keras.utils.set_random_seed(0)

        chars_oh = tf.expand_dims(self.config.EMBED.to_onehot(prompt), 0)
        self.print_1(f"# [TextGeneration - predict]:\tprompt_len={len(prompt)}\tnum_chars={pred_len}",
                     verbosity=verbosity)
        self.print_2("chars_oh shape:", chars_oh.shape, verbosity=verbosity)
        self.print_2("prompt_len:", len(prompt), verbosity=verbosity)

        chars_oh = self.pred_iter(chars_oh, pred_len, sampling_method, temp, True, verbosity)

        if not include_prompt:
            chars_oh = chars_oh[:, len(prompt):, :]

        self.print_3("chars_oh shape:", chars_oh.shape, verbosity=verbosity)
        if output_as == "string":
            return self.config.EMBED.to_string(chars_oh)
        elif output_as == "tensor":
            return self.config.EMBED.to_tfstring(chars_oh)

        return chars_oh

    def pred_iter(self, chars_oh, pred_len, sampling_method, temp, reset_states, verbosity):
        if reset_states:
            self.model_pred.reset_states()
        gen_loop = tqdm(range(pred_len), desc="Loading prompt...") if verbosity else range(pred_len + 1)
        logits = self.model_pred.predict(chars_oh, batch_size=1)[None, :, -1]
        for _ in gen_loop:
            sampled_id = sampling_method(logits=logits, temp=temp)
            self.print_2(f"\t => SAMPLED: {self.config.EMBED.to_tfstring(sampled_id)}", verbosity=verbosity)
            if verbosity:
                gen_loop.set_description("Generating...")
                gen_loop.set_postfix({"char": self.config.EMBED.to_tfstring(sampled_id).numpy()})

            sampled_oh = self.config.EMBED.to_onehot(sampled_id)
            self.print_3(f"sampled_oh shape: {sampled_oh.shape}", verbosity=verbosity)
            chars_oh = tf.concat([chars_oh, sampled_oh], axis=1)
            logits = self.model_pred.predict(sampled_oh, batch_size=1)
        return chars_oh

    def train(self, data: TextDataset, config: TrainingConfig = None, callbacks=None):
        # sourcery no-metrics
        self.print_1(f"# [TextGeneration - train]:\t{config}")
        if config is None:
            config = self.config.TRAINING

        SAVE_DIR, EPOCHS, BATCH_SIZE, BUFFER_SIZE = config.SAVE_DIR, config.EPOCHS, config.BATCH_SIZE, config.BUFFER_SIZE
        PRED_EVERY, PRED_LEN, PRED_TEMP = config.PRED_EVERY, config.PRED_LEN, config.PRED_TEMP

        data_shuf, data_batches = data.create_batches(BATCH_SIZE, BUFFER_SIZE)

        basedir = os.path.join(SAVE_DIR, data.label)
        callbacks, tb_file_writer = ModelUtils.create_callbacks(basedir, self.model_train, defaults=callbacks)

        epoch, es_cnt, es_delta, es_patience, es_monitor, history, sy_gen = 0, 0, 0.001, 10, 'loss', None, None
        tr_loop = tqdm(range(EPOCHS))
        self.history = {
            'loss': [],
            'acc': [],
            'best_loss': [],
            'gen': []
        }

        self.model_pred.reset_states()
        for epoch in tr_loop:
            history = self.model_train.fit(data_batches, epochs=1,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           use_multiprocessing=True,
                                           workers=16,
                                           callbacks=callbacks, verbose=0)
            self.model_train.reset_states()
            # tb_callback.set_model(self.model_train)

            if epoch % PRED_EVERY == 0:
                self.model_pred.set_weights(self.model_train.get_weights())

                sx, sy, sy_gen = self.dataset_sample_predict(dataset=data_shuf, pred_len=PRED_LEN,
                                                             temp=PRED_TEMP)[0]

                with tb_file_writer.as_default():
                    tf.summary.text("training", sy_gen, step=epoch)
            # save histories
            self.history['loss'].append(history.history['loss'][-1])
            self.history['acc'].append(history.history['acc'][-1])
            # should_save = False
            res = inf
            if len(self.history['best_loss']) > 0:
                if history.history['loss'][-1] < self.history['best_loss'][-1]:
                    res = history.history['loss'][-1]
                    # should_save = True
                else:
                    res = self.history['best_loss'][-1]

            self.history['best_loss'].append(res)
            self.history['gen'].append(sy_gen.numpy())

            # Implement early stopping based on the min delta, patience and monitor
            if epoch > 0 and self.history[es_monitor][-2] - self.history[es_monitor][-1] > es_delta:
                es_cnt = 0
            else:
                es_cnt += 1

            postfix = {
                "loss": round(self.history['loss'][-1], 4),
                "acc": round(self.history['acc'][-1], 4),
                "best_loss": self.history['best_loss'][-1],
                "gen": self.history['gen'][-1],
                "es": f"{es_cnt}/{es_patience}"
            }
            tr_loop.set_postfix(postfix)

            self.tensorboard_log(postfix, tb_file_writer, epoch)

            # if should_save:
            #     self.model_pred.set_weights(self.model_train.get_weights())
            #     self.save_model(basedir)

            if es_cnt >= es_patience:
                print("=> Early stopping...")
                break

        self.model_pred.set_weights(self.model_train.get_weights())
        print("=" * 80)
        print("# Running a few predictions with different temperatures. Please wait...")
        print("=" * 80)
        for temp in [0.25, 0.5, 0.75, 1.0]:
            print("# Temperature:", temp)
            _ = self.dataset_sample_predict(dataset=data_shuf, pred_len=500, temp=temp,
                                            tb_file_writer=tb_file_writer, silent=False)[0]
            print("-" * 80)

        print("# DONE!")
        return self.history

    def tensorboard_log(self, dictionary, tb_file_writer, step):
        with tb_file_writer.as_default():
            for k, v in dictionary.items():
                if type(v) == float:
                    tf.summary.scalar(k, v, step=step)

    def dataset_sample_predict(self, dataset, pred_len, temp: float, num_samples: int = 1,
                               tb_file_writer=None, tb_tag_prefix="", silent=True):
        sample_predictions = []
        for x, y in dataset.take(num_samples):
            if not silent:
                print(f"Prompt: {self.config.EMBED.to_tfstring(x)}")
            gen = self.predict(x, Sampling.random_sampling, temp=temp, pred_len=pred_len,
                               output_as="tensor", verbosity=0)
            if not silent:
                print(f"Generated: {gen}")
            sample_predictions.append((x, y, gen))
            if tb_file_writer is not None:
                with tb_file_writer.as_default():
                    tf.summary.text(f"{tb_tag_prefix}gen_{temp}", gen, step=0)
        return sample_predictions

    def load_model(self, path, index=None):
        # dictionary of key-value pairs where the key is the foder name where a file named checkpoints.h5 exists
        # and the value is the path to the file
        models = {}
        for root, dirs, files in os.walk(path):
            if "checkpoints.h5" in files:
                print(f"Found model in {root}")
                models[os.path.basename(root)] = os.path.join(root, "checkpoints.h5")

        if not models:
            raise ValueError(f"No models found in {path}")

        options = list(models.keys())
        desc = "Select a model to load:\n"
        try:
            picked, _ = pick(options, desc, indicator='->')
        except Exception:
            if index is None:
                print("W: Could not show interactive menu, "
                      "please select a model manually by passing the index from the list below")
                for i, option in enumerate(options):
                    print(f"{i}: {option}")
                return
            else:
                print(f"W: Using index {index}")
                picked = options[index]

        model_path = models[picked]
        print(f"Loading model from {model_path}")
        self.model_train.load_weights(model_path)
        self.model_pred.load_weights(model_path)
