import tqdm
import pandas as pd
from datasets import load_metric
from .tokenize_utils import tokenize, get_n_random_sentences, lword_to_lstr, to_bleu_references
from .Sampling import generate_all_sampling
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
    def get_metrics(predictions, references):
        # tokenize the input texts and get reference texts
        predictions = list(tokenize(predictions))
        predict_len = len(predictions)
        n_references = get_n_random_sentences(references, predict_len)

        # get metrics
        metric_values = {}
        metric_values.update(bleu.compute(predictions=predictions, references=list(to_bleu_references(n_references))))
        metric_values.update(
            rouge.compute(predictions=lword_to_lstr(predictions), references=lword_to_lstr(n_references)))
        metric_values.update(perplexity.compute(input_texts=lword_to_lstr(predictions), model_id='gpt2'))

        return metric_values


def get_metrics(predictions, references):
    metric_values = {'predictions': predictions}

    # tokenize the input texts and get reference texts
    predictions = list(tokenize(predictions))
    predict_len = len(predictions)
    n_references = get_n_random_sentences(references, predict_len)

    # get metrics
    metric_values.update(bleu.compute(predictions=predictions, references=list(to_bleu_references(n_references))))
    metric_values.update(rouge.compute(predictions=lword_to_lstr(predictions), references=lword_to_lstr(n_references)))
    metric_values.update(perplexity.compute(input_texts=lword_to_lstr(predictions), model_id='gpt2'))

    return metric_values


def get_metrics_for_all_sampling(generator, prompt, references):
    greedy, random, top_k, top_p = generate_all_sampling(generator, prompt)
    return {
        'greedy': get_metrics(greedy, references),
        'random': get_metrics(random, references),
        'top_k': get_metrics(top_k, references),
        'top_p': get_metrics(top_p, references)
    }


def get_metrics_for_all_prompts(generator, prompts, references):
    metrics = {}
    for i, prompt in tqdm.tqdm(enumerate(prompts)):
        metrics[i] = get_metrics_for_all_sampling(generator, prompt, references)
    return metrics


def metric_dict_to_df(metrics):
    cols = ['prompt_id', 'sampling_type'] + list(metrics[0]['greedy'].keys())
    df = pd.DataFrame(columns=cols)
    for i, metric_dict in metrics.items():
        for sampling_type, metric_values in metric_dict.items():
            row = {'prompt_id': i, 'sampling_type': sampling_type}
            row.update(metric_values)
            df = df.append(row, ignore_index=True)
    return df
