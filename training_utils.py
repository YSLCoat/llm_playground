import torch
import torch.nn as nn

class Trainer():
    def __init__(self, configs, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, device, val_loader=None):
        self.configs = configs
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def process_batch(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        model_output = self.model(inputs, targets)
        loss = self.loss_fn(model_output.transpose(1, 2), targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        batch_loss = loss.item() / len(inputs)
        print(f"Batch Loss: {batch_loss}")
        return batch_loss

    def process_epoch(self, dataloader):
        epoch_loss = 0 
        for i, (inputs, targets) in enumerate(dataloader):
            epoch_loss += self.process_batch(inputs, targets)

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch Loss: {avg_epoch_loss}")

    def train(self):
        for epoch in range(self.configs.epochs):
            self.process_epoch(self.train_loader)

    
