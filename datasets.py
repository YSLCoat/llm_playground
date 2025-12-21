import pathlib

import torch
from torch.utils.data import Dataset


class TinyShakespeare(Dataset):
    def __init__(self, tokenizer, datadir):
        self.datadir = datadir
        self.data_path = pathlib.Path(self.datadir, "tinyshakespeare.txt")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.tokernizer = tokenizer(self.text)

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        raise NotImplementedError
    