## CS525 Final Project

## TODO:

- Find papers of smaller models to compare results
- Look into metric for the complexity of text (e.g. 5th grader vs 12th grader, etc)
	- Include this text comprehension metric in the paper
- https://www.tensorflow.org/tfx/tutorials/serving/rest_simple

# Ideas:
## Losses:
- Categorical cross-entropy?
- Cosine similarity?

## Tokenizers:
- We can use pre-trained tokenizers from: https://huggingface.co/docs/transformers/v4.18.0/en/fast_tokenizers
  - There's the default PreTrainedTokenizer and PreTrainedTokenizerFast
  - https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
  - https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
  - https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/tokenizer#transformers.BatchEncoding
- Types:
  - Subword tokenization: Bert and XLNet tokenizers.
  - Byte-Pair encoding: GPT-2 and Roberta.
- Vocab Size Comparison:
  - GPT: 40,478 (478 base characters and training was stopped after 40,000 merges)
  - GPT-2: 50,257 (forced base vocab Byte-level [256 chars] and done 50,000 merges)
  - Transformer XL: 267,735
- Links: 
  - https://huggingface.co/docs/transformers/tokenizer_summary
  - https://towardsdatascience.com/a-comprehensive-guide-to-subword-tokenisers-4bbd3bad9a7c
  - https://towardsdatascience.com/comparing-transformer-tokenizers-686307856955
  - https://arxiv.org/pdf/2204.08832.pdf

## Creating our custom architecture on top of HuggingFace
- https://huggingface.co/docs/transformers/main/en/create_a_model#model
- https://huggingface.co/docs/transformers/main/en/add_new_model

## Generation & Sampling:
- We can make our generation implement these:
  - https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
  - https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_tf_utils.TFGenerationMixin
  - supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling. 
- Sampling Methods
  - Random Search with Temperature
  - Beam Search - will likely perform better

## Loss Functions
- An Edit-Invariant Sequence Loss for Text Generation
  - https://arxiv.org/pdf/2106.15078.pdf

## Optimizers:
- Adam Weight Decay
  - https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.AdamWeightDecay
- Warmup Schedule
  - https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.WarmUp

## LM Architecture Comparison:
- Look through implementations of different models at: https://github.com/huggingface/transformers/tree/main/src/transformers
- Simplest to most complicated:
  - GPT: unidirectional
  - BERT: bidirectional (probability is based on a few masked words in a sentence)
    - text generation becomes a little more complicated because of this masking
  - XLNet: "probability of any sequence can be modeled using any permutation in an auto regressive fashion"
    - two-stream attention mechanism
    - introduces peculiarities during training (target and non-target tokens are handled differently)
- From: https://stackoverflow.com/questions/57845439/which-model-gpt2-bert-xlnet-and-etc-would-you-use-for-a-text-classification
  - GPT2 architecture works best on short-paragraph sized notes.
  - BERT performs better for longer texts (up to 2 pages).
  - XLNet probably beats them all.

## Data
- Doesn't matter rn we can test on anything and choose final data later

## Repos:
- openai/human-eval: Code for the paper "Evaluating Large Language Models Trained on Code"
- kimiyoung/transformer-xl
- openai/gpt-2: Code for the paper "Language Models are Unsupervised Multitask Learners"
