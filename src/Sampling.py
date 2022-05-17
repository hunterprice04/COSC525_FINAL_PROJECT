import tensorflow as tf


class SamplingEnums:
    """
    Enums for sampling methods
    """
    ALL = list(range(0, 6))
    GREEDY = 0
    BEAM = 1
    RANDOM = 2
    TOP_K = 3
    TOP_P = 4
    SAMPLE = 5


class Sampling:

    def __init__(self, model):
        self.model = model
        self.sample_dict = {
            'greedy': self.greedy,
            'beam_search': self.beam_search,
            'random': self.random,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'sample': self.sample
        }
        self.sample_list = list(self.sample_dict.values())

    def greedy(self, input_ids, max_length=50, **kwargs):
        return self.model.generate(
            input_ids,
            max_length=max_length,
            **kwargs)

    def beam_search(self,
                    input_ids,
                    max_length=50,
                    num_beams=5,
                    early_stopping=True,
                    **kwargs):
        return self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            **kwargs
        )

    def random(self, input_ids,
               seed=None,
               do_sample=True,
               max_length=50,
               temperature=0.01,
               **kwargs):
        if seed is not None:
            tf.random.set_seed(seed)
        return self.model.generate(
            input_ids,
            do_sample=do_sample,
            max_length=max_length,
            top_k=0,
            temperature=temperature,
            **kwargs
        )

    def top_k(self, input_ids,
              seed=None,
              do_sample=True,
              max_length=50,
              top_k=0,
              **kwargs):
        if seed is not None:
            tf.random.set_seed(seed)
        return self.model.generate(
            input_ids,
            do_sample=do_sample,
            max_length=max_length,
            top_k=top_k,
            **kwargs
        )

    def top_p(self, input_ids,
              seed=None,
              do_sample=True,
              max_length=50,
              top_p=0.92,
              **kwargs):
        if seed is not None:
            tf.random.set_seed(seed)
        return self.model.generate(
            input_ids,
            do_sample=do_sample,
            max_length=max_length,
            top_k=0,
            top_p=top_p,
            **kwargs
        )

    def sample(self, input_ids, **kwargs):
        """
        This allows the user to manually specify any type of sampling manually
        """
        return self.model.generate(input_ids, kwargs)


def generate_all_sampling(generator, prompt, n_tokens=50):
    greedy = generator.generate(prompt, n_tokens, generator.GREEDY)
    random = generator.generate(prompt, n_tokens, generator.RANDOM)
    top_k = generator.generate(prompt, n_tokens, generator.TOP_K)
    top_p = generator.generate(prompt, n_tokens, generator.TOP_P)
    return greedy, random, top_k, top_p
