import os
import pickle

from src import Utils
from src.Dataset import Dataset
from src.Generation import GenerationCallback, Generator
from src.SimpleTransformer import SimpleTransformer
from src.Tokenizer import Tokenizer


class SimpleTransformerLMHead:
    def __init__(self, config):
        print("# [SimpleTransformerLMHead] Initializing...")
        self.config = config
        # Read in the data and create the dataset
        self.dataset = Dataset(config.TRAINING.DATASET)
        # Create the tokenizer
        self.tokenizer = Tokenizer(self.dataset, model_config=config.MODEL)
        self.dataset_seq = self.dataset.create_batch_sequences(self.tokenizer, batch_sz=config.TRAINING.BATCH_SIZE)
        self.transformer_model = SimpleTransformer.create_model(config.MODEL)
        self.transformer_model.summary()
        self.generator = Generator(self.transformer_model, config.MODEL.MAX_LEN, self.tokenizer.vocab)

    def train(self, eval_prompt="i will always be", eval_prompt_len=100, override_epochs=None):
        print("# [SimpleTransformerLMHead] Training...")
        seq_len, vocab = self.config.MODEL.MAX_LEN, self.tokenizer.vocab
        epochs = self.config.TRAINING.EPOCHS
        if override_epochs is not None:
            epochs = override_epochs
        callbacks, tb_file_writer = Utils.create_callbacks("logs", self.transformer_model)
        gen_callback = GenerationCallback(eval_prompt, max_tokens=eval_prompt_len, seq_len=seq_len, vocab=vocab)
        callbacks.append(gen_callback)
        self.transformer_model.fit(self.dataset_seq, verbose=1, epochs=epochs, callbacks=callbacks)

    def save(self, dir_path="model_save"):
        self.transformer_model.save(dir_path)
        with open(os.path.join(dir_path, "vocab.pkl"), "wb") as f:
            pickle.dump(self.tokenizer.vocab, f)
        with open(os.path.join(dir_path, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)

    def generate(self, prompt, max_tokens=25):
        print("# [SimpleTransformerLMHead] Generating...")
        self.generator.generate(prompt, max_tokens)
