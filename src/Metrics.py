import tensorflow as tf
import itertools
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu


class Perplexity:
    @staticmethod
    def get_perplexity(reference_sentences, evaluated_sentences):
        """
        Calculate the perplexity. Perplexity is used as a measure of probability
                for a sentence to be produced by the model trained on a dataset.
        GOOD: we ultimately want to check perplexity values on the test set
                and choose the model with the lowest value for this metric.
        BAD: If the model is completely dumb (worst possible),
                perplexity = |v| i.e. size of the vocabulary.
        The lower the perplexity value, the better the model
        :param reference_sentences:
        :param evaluated_sentences:
        :return: The perplexity of the sentence.
        """
        # TODO: Implement
        print("Not implemented!")
        return None


class Bleu:

    @staticmethod
    def get_score(reference_sentences, evaluated_sentences):
        """
        calculate pair wise bleu score. uses nltk implementation
        Args:
            references : a list of reference sentences
            candidates : a list of candidate(generated) sentences
        Returns:
            bleu score(float)
        """
        ref_bleu, gen_bleu = [], []
        gen_bleu.extend(l.split() for l in evaluated_sentences)
        ref_bleu.extend([l.split()] for l in reference_sentences)
        cc = SmoothingFunction()
        return corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)


class Rouge:
    @staticmethod
    def _split_into_words(sentences):
        """Splits multiple sentences into words and flattens the result"""
        return list(itertools.chain(*[_.split(" ") for _ in sentences]))

    @staticmethod
    def _get_word_ngrams(n, sentences):
        """Calculates word n-grams for multiple sentences.
        """
        assert len(sentences) > 0
        assert n > 0

        words = Rouge._split_into_words(sentences)
        return Rouge._get_ngrams(n, words)

    @staticmethod
    def _get_ngrams(n, text):
        """Calcualtes n-grams.
        Args:
          n: which n-grams to calculate
          text: An array of tokens
        Returns:
          A set of n-grams
        """
        text_length = len(text)
        max_index_ngram_start = text_length - n
        return {tuple(text[i:i + n]) for i in range(max_index_ngram_start + 1)}

    @staticmethod
    def rouge_n(reference_sentences, evaluated_sentences, n=2):
        """
        Computes ROUGE-N of two text collections of sentences.
        Source: http://research.microsoft.com/en-us/um/people/cyl/download/
        papers/rouge-working-note-v1.3.1.pdf
        Args:
          evaluated_sentences: The sentences that have been picked by the summarizer
          reference_sentences: The sentences from the referene set
          n: Size of ngram. Defaults to 2.
        Returns:
          recall rouge score(float)
        Raises:
          ValueError: raises exception if a param has len <= 0
        """
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_ngrams = Rouge._get_word_ngrams(n, evaluated_sentences)
        reference_ngrams = Rouge._get_word_ngrams(n, reference_sentences)
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        # Handle edge case. This isn't mathematically correct, but it's good enough
        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        recall = 0.0 if reference_count == 0 else overlapping_count / reference_count
        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

        # just returning recall count in rouge, useful for our purpose
        return recall


class Metrics:
    @staticmethod
    def evaluate_metrics(reference_sentences, evaluated_sentences):
        """
        Evaluate metrics for Bleu, Rouge, and Perplexity
        :param reference_sentences:
        :param evaluated_sentences:
        :return:
        """
        bleu = Bleu()
        rouge = Rouge()
        perplexity = Perplexity()

        bleu_score = bleu.get_score(reference_sentences, evaluated_sentences)
        rouge_score = rouge.rouge_n(reference_sentences, evaluated_sentences)
        perplexity_score = perplexity.get_perplexity(reference_sentences, evaluated_sentences)

        return bleu_score, rouge_score, perplexity_score
