import torch
import torch.nn as nn
from model.modelFactory import ModelFactory
from torch import optim


class Model:
    def __init__(self, params):
        """Initialize the Model class with given parameters.
        
        Args:
            params (dict): Dictionary of parameters including model configurations.
        """
        self.params = params

    def apply_weights(self, y_train: torch.Tensor, weight_before_reversal=1, weight_after_reversal=2):
        """Apply custom weights to the training labels based on trend reversal.

        Args:
            y_train (torch.Tensor): The training labels.
            weight_before_reversal (int, optional): The weight before a trend reversal. Defaults to 1.
            weight_after_reversal (int, optional): The weight after a trend reversal. Defaults to 2.

        Returns:
            torch.Tensor: The computed weights for each label.
        """
        weights = torch.zeros_like(y_train)
        for idx, sub_y_train in enumerate(y_train):
            array = sub_y_train.numpy()
            sub_weights = [weight_before_reversal] * len(array)
            for i in range(1, len(array)):
                if not (array[i] == array[i-1]).all():
                    sub_weights[i:] = [weight_after_reversal] * (len(array) - i)
                    break
            weights[idx] = torch.tensor(sub_weights)
        return weights

    def get_criterion(self, apply_weights, weight_before_reversal, weight_after_reversal, y_batch):
        """Retrieve the appropriate loss function with optional weights.

        Args:
            apply_weights (bool): Whether to apply custom weights.
            weight_before_reversal (int): The weight before a trend reversal.
            weight_after_reversal (int): The weight after a trend reversal.
            y_batch (torch.Tensor): The batch of labels.

        Returns:
            nn.Module: The loss function (criterion).
        """
        if apply_weights:
            weights = self.apply_weights(y_batch, weight_before_reversal, weight_after_reversal)
            criterion = nn.BCEWithLogitsLoss(weight=weights)
        else:
            criterion = nn.BCEWithLogitsLoss()
        return criterion

    def run_training_epoch(self, data_loader, model, optimizer):
        """Run a single training epoch.

        Args:
            data_loader (DataLoader): DataLoader for the dataset.
            model (nn.Module): The neural network model to train.
            optimizer (torch.optim.Optimizer): The optimizer for updating model weights.

        Returns:
            float: Average loss for the epoch.
            float: Average accuracy for the epoch.
        """
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            criterion = self.get_criterion(self.params['apply_weights'], 
                                           self.params['weight_before_reversal'], 
                                           self.params['weight_after_reversal'], y_batch)
            loss = criterion(outputs, y_batch)
            acc = self.binary_accuracy(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            total_acc += acc.item() * X_batch.size(0)
        
        avg_loss = total_loss / len(data_loader.dataset)
        avg_acc = total_acc / len(data_loader.dataset)
        
        return model, avg_loss, avg_acc

    def run_validation_epoch(self, data_loader, model):
        """Run a single validation epoch.

        Args:
            data_loader (DataLoader): DataLoader for the validation dataset.
            model (nn.Module): The neural network model to evaluate.

        Returns:
            float: Average validation loss for the epoch.
            float: Average validation accuracy for the epoch.
        """
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                outputs = model(X_batch)
                criterion = self.get_criterion(self.params['apply_weights'], 
                                               self.params['weight_before_reversal'], 
                                               self.params['weight_after_reversal'], y_batch)
                loss = criterion(outputs, y_batch)
                acc = self.binary_accuracy(outputs, y_batch)
                
                total_loss += loss.item() * X_batch.size(0)
                total_acc += acc.item() * X_batch.size(0)
        
        avg_loss = total_loss / len(data_loader.dataset)
        avg_acc = total_acc / len(data_loader.dataset)
        
        return avg_loss, avg_acc

    def binary_accuracy(self, preds, y):
        """Calculate the binary accuracy of the predictions.

        Args:
            preds (torch.Tensor): The predictions made by the model.
            y (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The accuracy of the predictions.
        """
        rounded_preds = torch.round(torch.sigmoid(preds))  # Apply sigmoid function before rounding
        correct = (rounded_preds == y).float()
        acc = correct.sum() / correct.numel()
        return acc

    def early_stopping(self, val_loss, best_val_loss, epochs_no_improve, patience, rollback_epoch, model, best_model):
        """Checks if training should be stopped early and updates the best model if needed.
        
        Args:
            val_loss (float): Current validation loss.
            best_val_loss (float): Best validation loss observed so far.
            epochs_no_improve (int): Number of epochs with no improvement in validation loss.
            patience (int): Number of epochs to wait before early stopping.
            rollback_epoch (int): The epoch number to rollback to if early stopping is triggered.
            model (nn.Module): The current model being trained.
            best_model (OrderedDict): The state dictionary of the best model.

        Returns:
            bool: Whether to stop training early.
            int: Updated epochs_no_improve.
            int: Updated rollback_epoch.
            float: Updated best validation loss.
            OrderedDict: Updated best model state dictionary.
        """
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            rollback_epoch = rollback_epoch + 1
            best_model = model.state_dict()
        else:
            epochs_no_improve += 1

        early_stop = epochs_no_improve >= patience
        return early_stop, epochs_no_improve, rollback_epoch, best_val_loss, best_model

    def train_model(self, train_loader, val_loader, num_epochs=20, patience=5):
        """Train the model using the specified parameters and datasets.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            num_epochs (int, optional): Number of epochs for training. Defaults to 20.
            patience (int, optional): Number of epochs to wait before early stopping. Defaults to 5.

        Returns:
            dict: Training history including losses, accuracies, and rollback epoch (if any).
        """
        model_type = self.params['model_type']
        model = ModelFactory.create_model_instance(model_type, self.params)
        optimizer = optim.Adam(model.parameters(), lr=self.params['learning_rate'])
        
        history = {
            'loss': [],
            'binary_accuracy': [],
            'val_loss': [],
            'val_binary_accuracy': [],
            'rollback_epoch': None
        }
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        rollback_epoch = 0
        best_model = None
        
        for epoch in range(num_epochs):
            model, train_loss, train_acc = self.run_training_epoch(train_loader, model, optimizer)
            val_loss, val_acc = self.run_validation_epoch(val_loader, model)
            
            history['loss'].append(train_loss)
            history['binary_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_binary_accuracy'].append(val_acc)
            
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, \
                Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", file=open('log.txt', 'a'))
            
            # Check for early stopping
            early_stop, epochs_no_improve, rollback_epoch, best_val_loss, best_model = self.early_stopping(
                val_loss, best_val_loss, epochs_no_improve, patience, rollback_epoch, model, best_model)
            
            if early_stop:
                print(f'Early stopping at epoch {epoch + 1}', file=open('log.txt', 'a'))
                model.load_state_dict(best_model)
                history['rollback_epoch'] = rollback_epoch
                break

        return model, history

    def online_train_model(self, model, data_loader, num_epochs=20, patience=5):
        """Train the model online with new data.

        Args:
            model (nn.Module): The neural network model to train.
            data_loader (DataLoader): DataLoader providing the new data.
            num_epochs (int, optional): Number of epochs for training. Defaults to 20.
            patience (int, optional): Number of epochs to wait before early stopping. Defaults to 5.

        Returns:
            dict: Training history including losses, accuracies, and rollback epoch (if any).
        """
        optimizer = optim.Adam(model.parameters(), lr=self.params['learning_rate'])
        
        history = {
            'loss': [],
            'binary_accuracy': [],
            'rollback_epoch': None
        }
        
        best_loss = float('inf')
        epochs_no_improve = 0
        rollback_epoch = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            total_loss, total_acc = self.run_training_epoch(data_loader, model, optimizer)
            
            history['loss'].append(total_loss)
            history['binary_accuracy'].append(total_acc)
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Acc: {total_acc:.4f}", file=open('log.txt', 'a'))
            
            # Check for early stopping
            early_stop, epochs_no_improve, rollback_epoch, best_loss, best_model_state = self.early_stopping(
                total_loss, best_loss, epochs_no_improve, patience, rollback_epoch, model, best_model_state)
            
            if early_stop:
                print(f'Early stopping at epoch {epoch + 1}', file=open('log.txt', 'a'))
                model.load_state_dict(best_model_state)
                history['rollback_epoch'] = rollback_epoch
                break

        return model, history
