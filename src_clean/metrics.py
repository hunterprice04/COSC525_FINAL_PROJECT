from datasets import load_metric


# AVAILABLE METRICS
# https://huggingface.co/metrics

# https://github.com/huggingface/datasets/tree/master/metrics/bleu
bleu = load_metric("bleu")

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

        bleu_score = bleu.compute(predictions=evaluated_sentences, references=reference_sentences)
        rouge_score = rouge.compute(predictions=evaluated_sentences, references=reference_sentences)
        perplexity_score = perplexity.compute(input_texts=evaluated_sentences, model_id='gpt2')

        return bleu_score, rouge_score, perplexity_score