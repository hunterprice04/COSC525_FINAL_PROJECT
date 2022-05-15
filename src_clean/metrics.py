from datasets import load_metric


# AVAILABLE METRICS
# https://huggingface.co/metrics

# https://github.com/huggingface/datasets/tree/master/metrics/bleu
blue = load_metric("bleu")

# https://github.com/huggingface/datasets/tree/master/metrics/rouge
rouge = load_metric("rouge")

# https://github.com/huggingface/datasets/tree/master/metrics/perplexity
perplexity = load_metric("perplexity")



class Metrics:


    @staticmethod
    def evaluate_metrics(reference_sentences, evaluated_sentences):
        """
        Evaluate metrics for Bleu, Rouge, and Perplexity
        :param reference_sentences:
        :param evaluated_sentences:
        :return:
        """

        bleu_score = bleu.compute(reference_sentences, evaluated_sentences)
        rouge_score = rouge.compute(reference_sentences, evaluated_sentences)
        perplexity_score = perplexity.compute(reference_sentences, evaluated_sentences)

        return bleu_score, rouge_score, perplexity_score