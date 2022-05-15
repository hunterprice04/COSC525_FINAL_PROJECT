import re
from random import choices


def gen_n_sent(f, n):
    return list(flatten([f() for _ in range(n)]))


def lword_to_lstr(sentences):
    return (' '.join(s) for s in sentences)


def pprint_sentences(sentences):
    for i, s in enumerate(lword_to_lstr(sentences)):
        print('-' * 100)
        print("{}: {}".format(i, s))


def get_n_random_sentences(sentences, n):
    return choices(sentences, k=n)


def filter_empty_sentences(sentences):
    return (sentence for sentence in sentences if len(sentence) > 0)


def nest_sentences(sentences):
    return ([sentence] for sentence in sentences)


def filter_empty_words(sentences):
    return ([w for w in s if len(w) > 0] for s in sentences)


def flatten(iterables):
    return (elem for iterable in iterables for elem in iterable)


def clean_sentences(sentences):
    return (re.sub(' +', ' ', sentence.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')) for sentence in
            sentences)


def split_sentences(sentences):
    return flatten((s.split('.') for s in sentences))


def split_words(sentences):
    return (sentence.split(' ') for sentence in sentences)


def tokenize(sentences):
    return filter_empty_sentences(
        filter_empty_words(
            split_words(
                filter_empty_sentences(
                    split_sentences(
                        clean_sentences(sentences))))))


def to_bleu_references(sentences):
    return nest_sentences(sentences)


def tokenize_prompts(prompts, tokenizer):
    return {i: tokenizer.encode(prompt, return_tensors='tf') for i, prompt in enumerate(prompts)}
