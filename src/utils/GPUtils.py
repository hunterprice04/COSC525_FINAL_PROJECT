# Adapted from: https://gist.github.com/s-mawjee/ad0d8e0c7e07265cae097899fe48c023
import nvidia_smi

_GPU = False
_NUMBER_OF_GPU = 0


class GPUtils:

    @staticmethod
    def print_usage():
        global _GPU
        _GPU = GPUtils._check_gpu()
        if _GPU:
            GPUtils._print_gpu_usage()
        else:
            print("No GPU found.")

    @staticmethod
    def _check_gpu():
        global _NUMBER_OF_GPU
        nvidia_smi.nvmlInit()
        _NUMBER_OF_GPU = nvidia_smi.nvmlDeviceGetCount()
        if _NUMBER_OF_GPU > 0:
            return True
        return False

    @staticmethod
    def _print_gpu_usage(detailed=False, prefix: str = ''):
        global _NUMBER_OF_GPU
        for i in range(_NUMBER_OF_GPU):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            if not detailed:
                print(f'GPU-{i}: {GPUtils._bytes_to_megabytes(info.used)}/'
                      f'{GPUtils._bytes_to_megabytes(info.total)} MB')
            else:
                print(prefix + "({:.2f}% free): {:.2f}GB (total), {:.2f}GB (free), {:.2f}GB (used)".format(
                    100 * info.free / info.total,
                    info.total / 1024 / 1024 / 1024,
                    info.free / 1024 / 1024 / 1024,
                    info.used / 1024 / 1024 / 1024
                ))

    @staticmethod
    def _bytes_to_megabytes(bytes):
        return round((bytes / 1024) / 1024, 2)
