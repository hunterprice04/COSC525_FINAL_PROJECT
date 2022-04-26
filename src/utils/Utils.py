import os

import nvidia_smi
from transformers import AutoTokenizer, TFAutoModelForCausalLM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Utils:

    @staticmethod
    def print_mem(prefix: str = ''):
        nvidia_smi.nvmlInit()

        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print(prefix + "({:.2f}% free): {:.2f}GB (total), {:.2f}GB (free), {:.2f}GB (used)".format(
                100 * info.free / info.total,
                info.total / 1024 / 1024 / 1024,
                info.free / 1024 / 1024 / 1024,
                info.used / 1024 / 1024 / 1024
            ))

        nvidia_smi.nvmlShutdown()

    @staticmethod
    def generate(model, tokenizer, text, max_length=512, temperature=0.7, top_k=50, top_p=0.9, print_mem=True):
        model, tokenizer = Utils.get_model_and_tokanizer(model, tokenizer, print_mem)
        if print_mem:
            Utils.print_mem("# MEM BEFORE: ")
        # Generate a sequence of tokens
        input_ids = tokenizer.encode(text, return_tensors='pt')

    @staticmethod
    def print_outputs(outputs, strip=True):
        print("=" * 80)
        print(f"# OUTPUTS: len={len(outputs)}")
        for i in range(len(outputs)):
            oup = outputs[i]
            if strip:
                oup = oup.split("\n")
                oup = [x for x in oup if x != '']
                oup = "\n".join(oup)
                # oup = oup.strip()
            print("-" * 80)
            print(f"# OUTPUT[{i}]: len={len(oup)}")
            print(oup)
