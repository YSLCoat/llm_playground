from datetime import datetime 

import torch
import torch.nn as nn

class Trainer():
    def __init__(self, configs, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, train_dataset, device, val_loader=None):
        self.configs = configs
        self.model = model
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def process_batch(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        model_output = self.model(inputs)
        loss = self.loss_fn(model_output.transpose(1, 2), targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        batch_loss = loss.item() #/ len(inputs)
        print(f"Batch Loss: {batch_loss}")
        return batch_loss

    def process_epoch(self, dataloader):
        epoch_loss = 0 
        for i, (inputs, targets) in enumerate(dataloader):
            epoch_loss += self.process_batch(inputs, targets)

        avg_epoch_loss = epoch_loss #/ len(dataloader)
        print(f"Epoch Loss: {avg_epoch_loss}")

    def train(self):
        for epoch in range(self.configs.epochs):
            self.process_epoch(self.train_loader)

        model_id = generate_model_id()
        self.save_model_checkpoint(model_id, 0)

    def save_model_checkpoint(self, run_id, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'configs': self.configs,
            'tokenizer': self.train_dataset.tokenizer
        }
        save_path = f"{run_id}_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

        

def generate_model_id(prefix="model"):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    return f"{prefix}_{timestamp}"

    
