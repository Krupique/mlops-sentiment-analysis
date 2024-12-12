# Training and evaluation logic
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # Defines the optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), lr=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = config['train']['num_epochs']
        self.device = device
        
        for param in list(self.model.transformer.parameters())[:3]:
            param.requires_grad = False 

        print(model)

        # Move the model to GPU
        self.model.to(self.device)

    def train(self):
        self.train_losses = []
        self.val_losses = []

        # Train loop
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            all_preds = []
            all_labels = []

            # Progress_bar
            train_loader_tqdm = tqdm(self.train_loader, desc=f'Training - Epoch {epoch+1}/{self.num_epochs}')

            for batch in train_loader_tqdm:
                input_ids, labels = batch
                # Move to GPU
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # Converting predictions to classes (index)
                preds = torch.argmax(outputs, dim=1)

                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())

                # Update progress bar with current average loss
                train_loader_tqdm.set_postfix({'Loss (batch)': loss.item()})

            # Average loss in training
            train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(train_loss)

            # Validation loop
            self.model.eval()
            val_running_loss = 0.0
            all_val_preds = []
            all_val_labels = []

            with torch.no_grad():
                for batch in self.valid_loader:
                    input_ids, labels = batch
                    # Move to GPU
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(input_ids)
                    val_loss = self.criterion(outputs, labels)
                    val_running_loss += val_loss.item()

                    # Converting predictions to classes (index)
                    val_labels = torch.argmax(labels, dim=1)
                    val_preds = torch.argmax(outputs, dim=1)

                    # Store predictions and labels
                    all_val_preds.extend(val_preds.cpu().numpy())
                    all_val_labels.extend(val_labels.cpu().numpy())
            # 
            val_loss_avg = val_running_loss / len(self.valid_loader)
            self.val_losses.append(val_loss_avg)

            # Calculate validation metrics  
            val_accuracy = accuracy_score(all_val_labels, all_val_preds)
            val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')

            # Print metrics after each epoch
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            print(f"Loss of Training: {train_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Precision: {val_precision:.4f}")
        