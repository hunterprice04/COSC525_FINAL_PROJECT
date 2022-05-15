from datasets import load_dataset


class HFDataset:
    @staticmethod
    def get_dataset(name, configuration, print_info=False, **kwargs):
        d = load_dataset(name, configuration, **kwargs)
        if print_info:
            print(f"# Dataset: {name} => {configuration}")
            try:
                print("\tShape:")
                for c, v in d.shape.items():
                    print(f"\t\t* {c}: {v}")
                print("\tColumn Names:")
                for k, v in d.column_names.items():
                    print(f"\t\t* {k}: {v}")
            except AttributeError:
                print(f"\t\t* {d.shape}")
                print("\tColumn Names:")
                print(f"\t\t* {d.column_names}")

        return d

    @staticmethod
    def print_samples(dataset, subset='train', col='text', n=5):
        for i, x in enumerate(dataset.data[subset][col][:n]):
            print("=" * 80)
            print(f"# Sample {i}:")
            print("-" * 40)
            print(x)
