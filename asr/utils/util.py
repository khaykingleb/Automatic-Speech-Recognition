import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def ensure_dir(dir_name):
    dir_name = Path(dir_name)
    if not dir_name.is_dir():
        dir_name.mkdir(parents=True, exist_ok=False)

def read_json(file_name):
    file_name = Path(file_name)
    with file_name.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, file_name):
    file_name = Path(file_name)
    with file_name.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    Setup GPU device if available. Get gpu device indices which are used for DataParallel.
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There's no GPU available on this machine, "
              "training will be performed on CPU.")
        n_gpu_use = 0

    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
               "available on this machine.")
        n_gpu_use = n_gpu
        print(f"Training will be performed on {n_gpu_use} GPUs.")
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:

    def __init__(self, *keys, writer=None):
        self.writer = writer
        self.data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self.data.columns:
            self.data[col].values[:] = 0

    def update(self, key, value, n=1):
        # TODO: Тут возможно ошибка
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self.data.total[key] += value * n
        self.data.counts[key] += n
        self.data.average[key] = self.data.total[key] / self.data.counts[key]

    def avg(self, key):
        return self.data.average[key]

    def result(self):
        return dict(self.data.average)

    def keys(self):
        return self.data.total.keys()
