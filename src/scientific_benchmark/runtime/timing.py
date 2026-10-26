"""Wall-clock timing; optional synchronized CUDA peak allocated memory."""
from time import perf_counter

class Timer:
    def __init__(self, *, cuda_device=None):
        self.cuda_device = cuda_device
        self.seconds = None
        self.peak_gpu_memory_bytes = None

    def __enter__(self):
        self.torch = None
        if self.cuda_device is not None:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA timing explicitly requested but CUDA unavailable')
            self.torch = torch
            torch.cuda.synchronize(self.cuda_device)
            torch.cuda.reset_peak_memory_stats(self.cuda_device)
        self.start = perf_counter()
        return self

    def __exit__(self, *exc):
        if self.torch is not None:
            self.torch.cuda.synchronize(self.cuda_device)
            self.peak_gpu_memory_bytes = int(self.torch.cuda.max_memory_allocated(self.cuda_device))
        self.seconds = perf_counter()-self.start
