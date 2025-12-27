import pathlib

import torch
from torch.utils.data import Dataset


class TinyShakespeare(Dataset):
    def __init__(self, text, tokenizer_cls, vocab=None, context_len=128):
        self.tokenizer = tokenizer_cls(text, vocab=vocab)
        self.context_len = context_len

    def __len__(self):
        return len(self.tokenizer.tokens) - self.context_len
    
    def __getitem__(self, idx):
        chunk = self.tokenizer.tokens[idx : idx + self.context_len + 1]
        inputs = chunk[:-1]
        targets = chunk[1:]
        return inputs, targets

    @classmethod
    def prepare_splits(cls, datadir, tokenizer_cls, train_ratio=0.9):
        data_path = pathlib.Path(datadir, "tinyshakespeare.txt")     
        with open(data_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        split_idx = int(len(full_text) * train_ratio)
        train_text = full_text[:split_idx]
        val_text = full_text[split_idx:]
        train_dataset = cls(train_text, tokenizer_cls, vocab=None)
        learned_vocab = train_dataset.tokenizer.character_to_integer
        val_dataset = cls(val_text, tokenizer_cls, vocab=learned_vocab)
        
        return train_dataset, val_dataset