import torch
import torch.nn as nn
from torch.utils.data import random_split

from training_utils import Trainer
from datasets import TinyShakespeare
from tokenizers import CharacterLevelTokenizer
from transformer import Transformer

from input_parser import parse_input_to_configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(configs):
    train_dataset, val_dataset = TinyShakespeare.prepare_splits(
        configs.datadir, 
        CharacterLevelTokenizer, 
        train_ratio=0.9
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)
    model = Transformer(1, train_dataset.tokenizer.vocab_size, 256, 0, 6, 8, configs.context_len, use_cross_attention=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), configs.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(configs, model, optimizer, loss_fn, train_loader, device, val_loader)
    trainer.train()

if __name__=="__main__":
    configs = parse_input_to_configs()
    main(configs)