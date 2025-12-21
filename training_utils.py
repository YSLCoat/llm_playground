import torch
import torch.nn as nn

class Trainer():
    def __init__(self, configs, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, val_loader=None):
        self.configs = configs
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def process_batch(self, inputs, targets):
        model_output = self.model(inputs)
        loss = self.loss_fn(model_output, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def process_epoch(self, dataloader):
        for i, (inputs, targets) in enumerate(dataloader):
            self.process_batch(inputs, targets)
        raise NotImplementedError

    def train(self):
        for epoch in range(self.configs.epochs):
            self.process_epoch(self.train_loader)

    
