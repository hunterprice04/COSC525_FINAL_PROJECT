from src.ModelUtilspy import ModelUtils


class HFModel:
    def __init__(self, model_name, tokenizer_name=None, print_mem=True):
        self.model, self.tokenizer = ModelUtils.get_model_and_tokanizer(model_name, tokenizer_name, print_mem)

    def generate(self, prompt=None, decode_outputs=True, **kwargs):
        """
        ====================================================
        PARAMS OF GENERATE:
        input_ids=None,
        max_length=None, min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None, top_k=None, top_p=None,
        repetition_penalty=None, length_penalty=None,
        bad_words_ids=None,
        bos_token_id=None, pad_token_id=None, eos_token_id=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
        use_cache=None,
        :param prompt:
        :param decode_outputs:
        :param kwargs:
        :return:
        """
        print("=" * 80)
        print("# GENERATE:")

        if prompt is None:
            input_ids = None
            print("No initial prompt, generating random sequence.")
        else:
            input_ids = self.encode_prompt(prompt)

        outputs = self.model.generate(
            input_ids=input_ids,
            **kwargs
        )
        print("-" * 80)
        print(f"Outputs shape: {outputs.shape}")

        if not decode_outputs:
            print("=" * 80)
            return outputs

        decoded_outputs = []
        for i in range(outputs.shape[0]):
            decoded = self.tokenizer.decode(outputs[i], skip_special_tokens=True, **kwargs)
            decoded = decoded.strip()
            decoded_outputs.append(decoded)
        return decoded_outputs

    def encode_prompt(self, prompt):
        print("-" * 80)
        print(f"Encoding prompt: len={len(prompt)}")
        print(prompt)
        return self.tokenizer.encode(prompt, return_tensors="tf")
