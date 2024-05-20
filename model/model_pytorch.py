import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# TODO: 調整變數名稱，將常數以大寫表示
class EarlyStopper:
    def __init__(self, patience=int(3), min_delta=float(0.01)):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, feature, type='loss'):
        """
        Checks if early stopping criteria is met.

        Args:
            validation_loss (float): The validation loss.

        Returns:
            bool: True if early stopping criteria is met, False otherwise.
        """
        if type == 'loss':
            if feature < self.min_validation_loss:
                self.min_validation_loss = feature
                self.counter = 0
            elif feature > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
        elif type == 'accuracy':
            if feature > self.min_validation_loss:
                self.min_validation_loss = feature
                self.counter = 0
            elif feature < (self.min_validation_loss - self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
    
class ModelBase(object):
    def _train_model(self):
        """
        Trains the model.

        Raises:
            NotImplementedError: Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _infer_model(self):
        """
        Infers the model.

        Raises:
            NotImplementedError: Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _online_training_model(self):
        """
        Performs online training of the model.

        Raises:
            NotImplementedError: Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

class NeuralNetworkModelBase(ModelBase):
    def __init__(self, params):
        """
        Initializes the ModelPyTorch class.

        Args:
            params (dict): A dictionary containing the parameters for the model.
        """
        self.params = params

    def _train_model(self, model, X_train, y_train, X_val, y_val, apply_weight):
        """
        Trains the model.
        """
        if apply_weight == 'True':
            train_weights = self.apply_weights(y_train, self.params['weight_before_reversal'], self.params['weight_after_reversal'])
            val_weights = self.apply_weights(y_val, self.params['weight_before_reversal'], self.params['weight_after_reversal'])
            # TODO: add function to change loss_function
            train_loss_function = nn.BCELoss(weight=train_weights)
            val_loss_function = nn.BCELoss(weight=val_weights)
        else:  
            train_loss_function = nn.BCELoss()
            val_loss_function = nn.BCELoss()
            
        # TODO: add function to change optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])
        early_stopper = EarlyStopper(patience=self.params['patience'], min_delta=self.params['min_delta']) 

        train_losses = []
        train_accuracy = []
        val_losses = []
        val_accuracy = []

        num_epochs = self.params['training_epoch_num']
        for epoch in tqdm(range(num_epochs)):
            model.train()
            optimizer.zero_grad()

            # forward pass
            outputs = model(X_train)
            loss = train_loss_function(outputs, y_train)
            # backward pass and update weights
            loss.backward()
            optimizer.step()

            # calculate training accuracy
            _, predicted = torch.max(outputs.data, -1)
            correct = (predicted == y_train.argmax(dim=-1)).sum().item()
            accuracy = correct / (y_train.size(-3)*y_train.size(-2))
            train_losses.append(loss.item())
            train_accuracy.append(accuracy)

            # calculate validation loss
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = val_loss_function(val_outputs, y_val)
                _, val_predicted = torch.max(val_outputs.data, -1)
                val_correct = (val_predicted == y_val.argmax(dim=-1)).sum().item()
                accuracy = val_correct / (y_val.size(-3)*y_val.size(-2))
                val_losses.append(val_loss.item())
                val_accuracy.append(accuracy)

            # early stopping based on training loss
            if early_stopper.early_stop(val_loss.item(), type='loss'):             
                break

        history = {
            'loss': train_losses,
            'binary_accuracy': train_accuracy,
            'val_loss': val_losses,
            'val_binary_accuracy': val_accuracy
        }
        return history, model

    def _infer_model(self, model, X_test):
        """
        Infers the model.

        Args:
            model: The PyTorch model.
            X_test: The input test data.

        Returns:
            The predicted values.
        """
        y_pred = model(X_test)
        return y_pred

    def _online_train_model(self, model, X_train, y_train, single_X_test, 
                        single_y_test, apply_weight, data_update_mode='append'):
        # Update the training dataset with the new instance
        if data_update_mode == 'append':
            online_X_train = torch.cat((X_train, single_X_test), dim=0)
            online_y_train = torch.cat((y_train, single_y_test), dim=0)
        elif data_update_mode == 'replace':
            online_X_train = torch.cat((X_train[1:], single_X_test), dim=0)
            online_y_train = torch.cat((y_train[1:], single_y_test), dim=0)
        elif data_update_mode == 'newest':
            online_X_train = single_X_test
            online_y_train = single_y_test
        else:
            raise ValueError(f"Invalid data update mode: {data_update_mode}")

        # Add the instance and its actual result to the training dataset
        X_train = np.append(X_train, single_X_test, axis=0)
        y_train = np.append(y_train, single_y_test, axis=0)
        
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        if apply_weight == 'True':
            online_train_weights = self.apply_weights(y_train, self.params['weight_before_reversal'], self.params['weight_after_reversal'])
            loss_function = nn.BCELoss(weight=online_train_weights)
        else:
            loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params['online_train_learning_rate'], weight_decay=self.params['online_train_weight_decay'])
        num_epochs = 10
        history = {}
        for epoch in range(num_epochs):
            # Retrain the model on this updated dataset
            model.train()
            optimizer.zero_grad()

            # forward pass
            outputs = model(online_X_train)
            loss = loss_function(outputs, online_y_train)
            _, predicted = torch.max(outputs.data, -1)
            correct = (predicted == online_y_train.argmax(dim=-1)).sum().item()
            accuracy = correct / (online_y_train.size(-3)*online_y_train.size(-2))
            # backward pass and update weights
            loss.backward()
            optimizer.step()
        history = {
            'loss': loss.item() / online_y_train.size(-2),
            'binary_accuracy': accuracy
            }
        return history, model
    
    def apply_weights(self, y_train: torch.tensor, weight_before_reversal=1, weight_after_reversal=2):
        weights = torch.zeros_like(y_train)
        for idx, sub_y_train in enumerate(y_train):
            array = sub_y_train.numpy()
            sub_weights = [weight_before_reversal] * len(array)
            for i in range(1, len(array)):
                if not (array[i] == array[i-1]).all():
                    sub_weights[i:] = [weight_after_reversal] * (len(array) - i)
                    break
            for j in range(len(sub_weights)):
                weights[idx, j] = torch.tensor([sub_weights[j]] * y_train.shape[2])
        return weights
       
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

class DummyClassifierModelBase(ModelBase):
    def __init__(self, params, input_shape):
        super(DummyClassifierModelBase, self).__init__()
        self.params = params
        
    def _train_model(self, model, X_train, y_train, X_val, y_val, apply_weight):
        label_counts = Counter(y_train)
        self.most_common_label = label_counts.most_common(1)[0][0]

        history = {
            'loss': [],
            'binary_accuracy': [],
            'val_loss': [],
            'val_binary_accuracy': []
        }
        return history, model

    def _infer_model(self, model, X_test):
        batch_size = X_test.size(0)
        predictions = torch.randint(0, 2, (batch_size, self.params['predict_steps'], 2), dtype=torch.float32)
        return predictions

    def _online_train_model(self, model, X_train, y_train, single_X_test, 
                        single_y_test, apply_weight, data_update_mode='append'):
        history = {
            'loss': [],
            'binary_accuracy': []
            }
        return history, model
 
class ModelLeNet(nn.Module, NeuralNetworkModelBase):
    def __init__(self, params=dict(), input_shape=tuple()):
        super(ModelLeNet, self).__init__()
        self.params = params

        # Convolution layers
        self.conv1 = nn.Conv1d(input_shape[1],
                               self.params["model_params"]["LeNet"]["conv_1_out_channels"],
                               kernel_size=self.params["model_params"]["LeNet"]["conv_1_kernel"],
                               padding=self.params["model_params"]["LeNet"]["conv_1_padding"])

        # Calculate size after convolutions and pooling
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool1d(self.params["model_params"]["LeNet"]["MaxPool2d_1_kernel_size"]),  # Will reduce each spatial dimension by half
        )
        self._get_conv_output((1, input_shape[1], self.params["look_back"]))

        # Fully connected layer
        self.fc1 = nn.Linear(self._to_linear, self.params["predict_steps"]*2)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(*shape)
            output = self.convs(input)
            self._to_linear = int(torch.flatten(output, 1).shape[1])

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 2, self.params["predict_steps"])  # Reshape to the desired output shape
        return x

class ModelLeNet_2layers(nn.Module, NeuralNetworkModelBase):
    def __init__(self, params=dict(), input_shape=tuple()):
        super(ModelLeNet_2layers, self).__init__()
        self.params = params

        # Convolution layers
        self.conv1 = nn.Conv1d(input_shape[1],
                               self.params["model_params"]["LeNet_2"]["conv_1_out_channels"],
                               kernel_size=self.params["model_params"]["LeNet_2"]["conv_1_kernel"],
                               padding=self.params["model_params"]["LeNet_2"]["conv_1_padding"])
        self.conv2 = nn.Conv1d(self.params["model_params"]["LeNet_2"]["conv_1_out_channels"],
                               self.params["model_params"]["LeNet_2"]["conv_2_out_channels"],
                               kernel_size=self.params["model_params"]["LeNet_2"]["conv_2_kernel"],
                               padding=self.params["model_params"]["LeNet_2"]["conv_2_padding"])

        # Calculate size after convolutions and pooling
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            nn.LayerNorm([self.params["model_params"]["LeNet_2"]["conv_1_out_channels"],
                          self.params["look_back"]]),
            nn.ReLU(),
            nn.MaxPool1d(self.params["model_params"]["LeNet_2"]["MaxPool2d_1_kernel_size"]),
            self.conv2,
            nn.LayerNorm([self.params["model_params"]["LeNet_2"]["conv_2_out_channels"],
                          int(self.params["look_back"]/self.params["model_params"]["LeNet_2"]["MaxPool2d_2_kernel_size"])]),
            nn.ReLU(),
            nn.MaxPool1d(self.params["model_params"]["LeNet_2"]["MaxPool2d_2_kernel_size"])
        )
        self._get_conv_output((1, input_shape[1], self.params["look_back"]))

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, self.params["model_params"]["LeNet_2"]["fc_1_out_features"])
        self.ln1 = nn.LayerNorm(self.params["model_params"]["LeNet_2"]["fc_1_out_features"])
        self.fc2 = nn.Linear(self.params["model_params"]["LeNet_2"]["fc_1_out_features"],
                             self.params["predict_steps"] * 2)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(*shape)
            output = self.convs(input)
            self._to_linear = int(torch.flatten(output, 1).shape[1])

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(-1, 2, self.params["predict_steps"])  # Reshape to the desired output shape
        return x

class ModelRNN(nn.Module, NeuralNetworkModelBase):
    def __init__(self, params=dict(), input_shape=tuple()):
        super(ModelRNN, self).__init__()
        self.params = params
        self.rnn = nn.RNN(input_size=input_shape[-1],
                          hidden_size=self.params["model_params"]["RNN"]["hidden_size"],
                          num_layers=self.params["model_params"]["RNN"]["num_layers"],
                          dropout=self.params["model_params"]["RNN"]["dropout"],
                          batch_first=True)
        self.lc = nn.LayerNorm([self.params["look_back"], self.params["model_params"]["RNN"]["hidden_size"]])
        self.fc = nn.Linear(self.params["model_params"]["RNN"]["hidden_size"], 2)

    def forward(self, x):
        # Forward pass through RNN
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.rnn(x, hidden)
        output = self.lc(output)
        # Reshape output to fit the fully connected layer
        output = output.contiguous().view(-1, self.params["model_params"]["RNN"]["hidden_size"])
        output = self.fc(output)
        # Reshape back to sequence format and align with target sequence length
        output = output.view(batch_size, -1, 2)  # [batch_size, sequence_length, output_size]
        output = torch.sigmoid(output)
        output = output[:, -self.params["predict_steps"]:, :]  # Take the last 'predict_steps' outputs
        return output

    def init_hidden(self, batch_size):
        # Initialize the hidden state
        return torch.zeros(self.params["model_params"]["RNN"]["num_layers"], batch_size, self.params["model_params"]["RNN"]["hidden_size"])

class ModelLSTM(nn.Module, NeuralNetworkModelBase):
    def __init__(self, params=dict(), input_shape=tuple()):
        super(ModelLSTM, self).__init__()
        self.params = params
        self.lstm = nn.LSTM(input_size=input_shape[-1],
                            hidden_size=self.params["model_params"]["LSTM"]["hidden_size"],
                            num_layers=self.params["model_params"]["LSTM"]["num_layers"],
                            dropout=self.params["model_params"]["LSTM"]["dropout"],
                            batch_first=True)
        self.fc = nn.Linear(self.params["model_params"]["LSTM"]["hidden_size"], 2)

    def forward(self, x):
        # Forward pass through LSTM
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        output, (hidden, cell) = self.lstm(x, hidden)
        # Reshape output to fit the fully connected layer
        output = output.contiguous().view(-1, self.params["model_params"]["LSTM"]["hidden_size"])
        output = self.fc(output)
        output = torch.sigmoid(output)
        # Reshape back to sequence format and align with target sequence length
        output = output.view(batch_size, -1, 2)  # [batch_size, sequence_length, output_size]
        output = output[:, -self.params["predict_steps"]:, :]  # Take the last 'predict_steps' outputs
        return output

    def init_hidden(self, batch_size):
        # Initialize the hidden state and cell state
        hidden_state = torch.zeros(self.params["model_params"]["LSTM"]["num_layers"], batch_size, self.params["model_params"]["LSTM"]["hidden_size"])
        cell_state = torch.zeros(self.params["model_params"]["LSTM"]["num_layers"], batch_size, self.params["model_params"]["LSTM"]["hidden_size"])
        return (hidden_state, cell_state)

# class ModelDNN_5layers(nn.Module, NeuralNetworkModelBase):
#     def __init__(self, params, input_shape):
#         super(ModelDNN_5layers, self).__init__()
#         self.params = params
#         # Calculate flattened input size
#         input_size = input_shape[1] * input_shape[2]
#         self.predict_steps = self.params['predict_steps']
#         # Fully connected layers
#         self.fc1 = nn.Linear(input_size, self.predict_steps*32)
#         self.fc2 = nn.Linear(self.predict_steps*32, self.predict_steps*16)
#         self.fc3 = nn.Linear(self.predict_steps*16, self.predict_steps*8)
#         self.fc4 = nn.Linear(self.predict_steps*8, self.predict_steps*4)
#         self.fc5 = nn.Linear(self.predict_steps*4, self.predict_steps*2)
#         # Layer normalization layers
#         self.ln1 = nn.LayerNorm(self.predict_steps*32)
#         self.ln2 = nn.LayerNorm(self.predict_steps*16)
#         self.ln3 = nn.LayerNorm(self.predict_steps*8)
#         self.ln4 = nn.LayerNorm(self.predict_steps*4)
#         self.ln5 = nn.LayerNorm(self.predict_steps*2)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         # Flatten the input
#         x = x.view(x.size(0), -1)  # Flatten to [batch_size, input_size]
#         # Fully connected layers with ReLU activations, layer normalization, and dropout
#         x = F.relu(self.ln1(self.fc1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.ln2(self.fc2(x)))
#         x = self.dropout(x)
#         x = F.relu(self.ln3(self.fc3(x)))
#         x = self.dropout(x)
#         x = F.relu(self.ln4(self.fc4(x)))
#         x = self.dropout(x)
#         x = self.ln5(self.fc5(x))  # Last layer with LayerNorm but without ReLU
#         # Apply sigmoid to the final layer
#         x = torch.sigmoid(x)
#         # Reshape the output to match target shape
#         output = x.view(-1, self.predict_steps, 2)  # Reshape to [batch_size, 2, 8]

#         return output


class ModelDNN_5layers(nn.Module, NeuralNetworkModelBase):
    def __init__(self, params, input_shape):
        super(ModelDNN_5layers, self).__init__()
        self.params = params
        # Calculate flattened input size
        input_size = input_shape[1] * input_shape[2]
        self.predict_steps = self.params['predict_steps']
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, self.predict_steps*32)
        self.fc2 = nn.Linear(self.predict_steps*32, self.predict_steps*16)
        self.fc3 = nn.Linear(self.predict_steps*16, self.predict_steps*8)
        self.fc4 = nn.Linear(self.predict_steps*8, self.predict_steps*4)
        self.fc5 = nn.Linear(self.predict_steps*4, self.predict_steps*2)

        # Layer normalization and dropout
        self.ln1 = nn.LayerNorm(self.predict_steps*32)
        self.ln2 = nn.LayerNorm(self.predict_steps*16)
        self.ln3 = nn.LayerNorm(self.predict_steps*8)
        self.ln4 = nn.LayerNorm(self.predict_steps*4)
        self.ln5 = nn.LayerNorm(self.predict_steps*2)
        self.dropout = nn.Dropout(self.params['model_params']["DNN_5layers"]['dropout'])

        # Apply Xavier Initialization to each linear layer
        self._init_weights()

    def _init_weights(self):
        # Use Xavier initialization for each linear layer
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)

        # Optional: Initialize biases with zeros or another method
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)
        nn.init.zeros_(self.fc5.bias)
        

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, input_size]
        # Fully connected layers with ReLU activations, layer normalization, and dropout
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.ln4(self.fc4(x)))
        x = self.dropout(x)
        # Final layer with LayerNorm but without ReLU
        x = self.ln5(self.fc5(x))
        # Apply sigmoid to the final layer
        x = torch.sigmoid(x)
        # Reshape the output to match target shape
        output = x.view(-1, self.predict_steps, 2)  # Reshape to [batch_size, self.predict_steps, 2]

        return output

class DummyClassifier(nn.Module, DummyClassifierModelBase):
    def __init__(self, params, input_shape):
        super(DummyClassifier, self).__init__()
        self.params = params
        self.predict_steps = self.params['predict_steps']

    def forward(self, x):
        # 獲取批次大小
        batch_size = x.size(0)
        # 隨機生成輸出，這裡假設輸出類別數為2，調整為需要的任何數量
        # 使用 torch.rand 生成介於 [0, 1) 的隨機數據，模擬隨機預測的結果
        random_output = torch.rand(batch_size, self.predict_steps, 2)
        return random_output

class ModelCNN_LSTM(nn.Module, NeuralNetworkModelBase):
    def __init__(self, params, input_shape):
        super(ModelCNN_LSTM, self).__init__()
        self.params = params
        self.feature_num = len(params["feature_cols"])
        
        # Convolution layers
        self.conv1 = nn.Conv1d(in_channels=self.feature_num,
                        out_channels=self.feature_num,
                        kernel_size=self.params["model_params"]["CNN_LSTM"]["conv_1_kernel"],
                        padding=self.params["model_params"]["CNN_LSTM"]["conv_1_padding"],
                        groups=self.feature_num)
        self.conv2 = nn.Conv1d(in_channels=self.feature_num,
                        out_channels=self.feature_num,
                        kernel_size=self.params["model_params"]["CNN_LSTM"]["conv_1_kernel"],
                        padding=self.params["model_params"]["CNN_LSTM"]["conv_1_padding"],
                        groups=self.feature_num)
                
        # Calculate size after convolutions and pooling
        self.convs = nn.Sequential(
            self.conv1,
            nn.LayerNorm([self.feature_num,
                          self.params["look_back"]]),
            nn.ReLU(),
            # nn.MaxPool1d(self.params["model_params"]["CNN_LSTM"]["MaxPool2d_1_kernel_size"]),
            self.conv2,
            nn.LayerNorm([self.feature_num,
                          int(self.params["look_back"])]),
            nn.ReLU(),
            # nn.MaxPool1d(self.params["model_params"]["CNN_LSTM"]["MaxPool2d_2_kernel_size"])
        )

        self.lstm = nn.LSTM(input_size=self.feature_num,
                    hidden_size=self.feature_num,
                    num_layers=self.params["model_params"]["CNN_LSTM"]["num_layers"],
                    dropout=self.params["model_params"]["CNN_LSTM"]["dropout"],
                    batch_first=True)
        
        # Fully connected layers
        self.fc = nn.Linear(self.feature_num, 2)
            
    def init_hidden(self, batch_size, x_size):
        # Initialize the hidden state and cell state
        hidden_state = torch.zeros(1, batch_size, self.feature_num)
        cell_state = torch.zeros(1, batch_size, self.feature_num)
        return (hidden_state, cell_state)
    
    def forward(self, x):
        x = x.view(-1, x.shape[-1], x.shape[-2])
        x = self.convs(x)
        x = x.view(-1, x.shape[-1], x.shape[-2])
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size, x.size())
        x, (hidden, cell) = self.lstm(x, hidden)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x[:, -self.params["predict_steps"]:, :]

        return x
        
class ModelFactory:
    @staticmethod
    def create_model_instance(model_type, params=None, input_shape=None):
        """
        Creates an instance of the specified model type.

        Args:
            model_type (str): The type of the model to create.
            params (dict): A dictionary containing the parameters for the model.
            input_shape (tuple): The shape of the input data.
            keras_model: The Keras model to be converted to PyTorch model.

        Returns:
            An instance of the specified model type.
        """
        models = {
            "LeNet": ModelLeNet,
            "LeNet_2": ModelLeNet_2layers,
            "RNN": ModelRNN,
            "LSTM": ModelLSTM,
            # "DNN_1layer": ModelDNN_1layer,
            # "DNN_3layers": ModelDNN_3layers,
            "DNN_5layers": ModelDNN_5layers,
            # "DNN_7layers": ModelDNN_7layers,
            "CNN_LSTM": ModelCNN_LSTM,
            # "seq2seq": Modelseq2seq,
            # "cnn": ModelCNN,
            # "fft": ModelFFT,
            # "ma": ModelMA,
            "DummyClassifier": DummyClassifier
            # Add other models here as needed
        }
        model_instance = models.get(model_type)
        if model_instance is None:
            raise ValueError(f"Invalid model type: {model_type}")
        instance = model_instance(params, input_shape)
        return instance

class Model:
    """A class representing a model for training and inference."""
    def __init__(self, params):
        """
        Initializes the ModelPyTorch class.

        Args:
            params (dict): A dictionary containing the parameters for the model.
        """
        self.params = params
        
    def create_model(self, model_type, input_shape=None):
        """Create a model instance.

        Args:
            model_type (str): The type of the model.
            params (dict, optional): The parameters for the model. Defaults to None.
            input_shape (tuple, optional): The shape of the input data. Defaults to None.

        Returns:
            model_instance: The created model instance.
        """
        model_instance = ModelFactory.create_model_instance(
            model_type, self.params, input_shape)
        return model_instance

    def train_model(self, model, X_train, y_train, X_val, y_val, apply_weight):
        """Train the model.

        Args:
            model: The model instance.
            X_train: The training input data.
            y_train: The training target data.

        Returns:
            The trained model.
        """
        return model._train_model(model, X_train, y_train, X_val, y_val, apply_weight)

    def infer_model(self, model, X_test):
        """Perform inference using the model.

        Args:
            model: The model instance.
            X_test: The input data for inference.

        Returns:
            The predicted output.
        """
        return model._infer_model(model, X_test)

    def online_train_model(self, model, X_train, y_train, single_X_test, 
                        single_y_test, apply_weight, data_update_mode):
        """Perform online training on the model.

        Args:
            model: The model instance.
            X_train: The training input data.
            y_train: The training target data.
            single_X_test: The input data for online training.
            single_y_test: The target data for online training.

        Returns:
            The updated model.
        """
        return model._online_train_model(model, X_train, y_train, single_X_test, 
                        single_y_test, apply_weight, data_update_mode)

    def run(self, X_train, y_train, X_test, y_test, X_val, y_val, pre_trained_model_path=None):
        """Run the model.

        Args:
            model_type (str): The type of the model.
            look_back (int): The number of previous time steps to consider.
            params (dict): The parameters for the model.
            X_train: The training input data.
            y_train: The training target data.
            X_test: The test input data.
            y_test: The test target data.
            pre_trained_model_path (str, optional): The path to a pre-trained model. Defaults to None.

        Returns:
            tuple: A tuple containing the trained model, training history, predicted outputs, 
            online training losses, and online training accuracy.
        """
        if pre_trained_model_path is not None:
            # model = torch.load(pre_trained_model_path)
            input_shape = X_train.shape
            model = self.create_model(self.params['model_type'], input_shape)
            model.load_state_dict(torch.load(pre_trained_model_path))
            history = None
        else:
            input_shape = X_train.shape
            model = self.create_model(self.params['model_type'], input_shape)
            history, model = self.train_model(model, X_train, y_train, X_val, y_val, self.params['apply_weight'])
            # torch.save(model, self.params['trained_model_path'])
            # torch.save(model.state_dict(), os.path.join(save_path_root, self.params['trained_model_path']))
        online_training_losses = []
        online_training_acc = []
        y_preds = []
        
        for i in tqdm(range(len(X_test)), file=open("log.txt", "a")):
            y_pred = self.infer_model(model, X_test[i:i+1])
            y_preds.append(y_pred[0])

            online_history, model = self.online_train_model(
                model, X_train, y_train, X_test[i:i+1], y_test[i:i+1], self.params['apply_weight'], self.params['data_update_mode'])
            online_training_losses.append(online_history['loss'])
            online_training_acc.append(
                online_history['binary_accuracy'])
        # torch.save(model, os.path.join(save_path_root, self.params['online_trained_model_path']))
        y_preds = torch.stack(y_preds).detach().numpy()

        return model, history, y_preds, online_training_losses, online_training_acc
