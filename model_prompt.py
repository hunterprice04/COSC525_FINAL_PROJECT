import sys
import pickle
from src.LMGenerator import LMGenerator

from tensorflow.python.keras.saving.save import load_model

sample_types = [
    'greedy', 'random', 'top_k', 'top_p'
]

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python model_prompt.py <sample_type> <max_tokens>")
        sys.exit(1)

    sample_type = sys.argv[1]
    max_tokens = int(sys.argv[2])

    if sample_type not in sample_types:
        print("Sample type must be one of: {}".format(sample_types))
        sys.exit(1)

    if max_tokens < 1:
        print("Max tokens must be greater than 1")
        sys.exit(1)

    # Load model from model.h5
    model = load_model('model_save')
    model_config = pickle.load(open('model_save/config_model.pkl', 'rb'))
    vocab = pickle.load(open('model_save/vocab.pkl', 'rb'))

    print('=' * 80)
    print('Starting interactive mode...')
    print('Type quit to exit.')
    print('=' * 80)
    prompt = input('> ')
    generator = LMGenerator(model, model_config.M_MAX_LEN, vocab)

    while prompt.strip() != 'quit':
        generated_txt = generator.generate(prompt, prompt, sample_type)
        print(generated_txt)
        prompt = input('> ')