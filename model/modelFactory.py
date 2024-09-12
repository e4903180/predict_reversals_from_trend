import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class NeuralNetworkModelBase(nn.Module, ABC):
    """
    Abstract base class for neural network models. 
    Ensures that all subclasses implement the `forward` method.
    """

    @abstractmethod
    def forward(self, x):
        """
        Abstract method that must be implemented by any subclass.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        pass


class GRU(NeuralNetworkModelBase):
    """GRU model for many-to-one sequence prediction.

    This model takes a sequence as input and predicts a single output.

    Args:
        params (dict): Dictionary of model parameters including the number of features, 
            look-back period, prediction steps, dropout rate, and the number of GRU layers.
    """
    
    def __init__(self, params):
        """Initializes the GRU model with the provided parameters."""
        super(GRU, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.dropout = params['dropout']  # Dropout rate for regularization
        self.num_layers = params['model_params']['GRU']['num_layers']  # Number of GRU layers
        
        # GRU layer with the specified number of features, layers, and dropout
        self.gru = nn.GRU(
            input_size=self.features_num, 
            hidden_size=self.features_num, 
            num_layers=self.num_layers, 
            batch_first=True, 
            dropout=self.dropout
        )
        
        # Fully connected layer that maps the GRU output to the required prediction steps
        self.fc = nn.Linear(self.features_num, self.predict_steps)

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take the output from the last time step
        out = self.fc(out)  # Apply the fully connected layer
        return out


class LSTM_many2many(NeuralNetworkModelBase):
    """LSTM model for many-to-many sequence prediction.

    This model takes a sequence as input and predicts a sequence of the same or different length.

    Args:
        params (dict): Dictionary of model parameters including the number of features, 
            look-back period, prediction steps, dropout rate, and the number of LSTM layers.
    """
    
    def __init__(self, params):
        """Initializes the LSTM_many2many model with the provided parameters."""
        super(LSTM_many2many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.dropout = params['dropout']  # Dropout rate for regularization
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        
        # LSTM layer with the specified number of features, layers, and dropout
        self.lstm = nn.LSTM(
            input_size=self.features_num, 
            hidden_size=self.features_num, 
            num_layers=self.num_layers, 
            batch_first=True, 
            dropout=self.dropout
        )
        
        # Fully connected layer that maps the LSTM output to the required prediction steps
        self.fc = nn.Linear(self.look_back * self.features_num, self.predict_steps)

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.lstm(x)
        out = out.reshape(out.size(0), -1)  # Flatten the LSTM output
        out = self.fc(out)  # Apply the fully connected layer
        return out


class LSTM(NeuralNetworkModelBase):
    """LSTM model for many-to-one sequence prediction.

    This model takes a sequence as input and predicts a single output.

    Args:
        params (dict): Dictionary of model parameters including the number of features, 
            look-back period, prediction steps, dropout rate, and the number of LSTM layers.
    """
    
    def __init__(self, params):
        """Initializes the LSTM model with the provided parameters."""
        super(LSTM, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.dropout = params['dropout']  # Dropout rate for regularization
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        
        # LSTM layer with the specified number of features, layers, and dropout
        self.lstm = nn.LSTM(
            input_size=self.features_num, 
            hidden_size=self.features_num, 
            num_layers=self.num_layers, 
            batch_first=True, 
            dropout=self.dropout
        )
        
        # Fully connected layer that maps the LSTM output to the required prediction steps
        self.fc = nn.Linear(self.features_num, self.predict_steps)

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the output from the last time step
        out = self.fc(out)  # Apply the fully connected layer
        return out


class AttentionLSTM(NeuralNetworkModelBase):
    """LSTM model with attention mechanism for sequence prediction.

    This model takes a sequence as input, applies attention, and predicts an output sequence.

    Args:
        params (dict): Dictionary of model parameters including the number of features, 
            look-back period, prediction steps, dropout rate, and the number of LSTM layers.
    """
    
    def __init__(self, params):
        """Initializes the AttentionLSTM model with the provided parameters."""
        super(AttentionLSTM, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.dropout = params['dropout']  # Dropout rate for regularization
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        
        # LSTM layer with the specified number of features, layers, and dropout
        self.lstm = nn.LSTM(
            input_size=self.features_num, 
            hidden_size=self.features_num, 
            num_layers=self.num_layers, 
            batch_first=True, 
            dropout=self.dropout
        )
        
        # Attention mechanism layer
        self.attention = nn.Linear(self.features_num, 1)
        
        # Fully connected layer that maps the attention-weighted LSTM output to the required prediction steps
        self.fc = nn.Linear(self.features_num, self.predict_steps)

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(out), dim=1)  # Compute attention weights
        out = torch.sum(out * attention_weights, dim=1)  # Apply attention weights to LSTM output
        out = self.fc(out)  # Apply the fully connected layer
        return out


class CNN(NeuralNetworkModelBase):
    """1D Convolutional Neural Network for sequence prediction.

    This model uses convolutional layers to process a sequence and predict future steps.

    Args:
        params (dict): Dictionary of model parameters including the number of features, 
            look-back period, prediction steps, dropout rate, and convolution settings.
    """
    
    def __init__(self, params):
        """Initializes the CNN model with the provided parameters."""
        super(CNN, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.dropout = params['dropout']  # Dropout rate for regularization
        
        # Convolutional layer with group convolution (depthwise convolution)
        self.conv1 = nn.Conv1d(
            in_channels=self.features_num, 
            out_channels=self.features_num, 
            kernel_size=3, 
            padding=1, 
            groups=self.features_num
        )
        
        # Max pooling layer to reduce the dimensionality of the sequence
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layer that maps the flattened convolutional output to the required prediction steps
        self.fc1 = nn.Linear(self.features_num * int(self.look_back / 2), self.predict_steps)  # Assuming input length is 64 and max pooling reduces it by half
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length) for Conv1D
        x = self.pool(F.relu(self.conv1(x)))  # Apply convolution, ReLU activation, and pooling
        x = self.dropout(x)  # Apply dropout
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)  # Apply the fully connected layer
        return x

class TransformerModel(NeuralNetworkModelBase):
    """Transformer-based model for sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including the number of features, 
            look-back period, prediction steps, and Transformer-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the TransformerModel with the provided parameters."""
        super(TransformerModel, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        input_dim = self.features_num
        hidden_dim = self.look_back 
        output_dim = self.predict_steps 
        num_layers = params['model_params']['Transformer']['num_layers']  # Number of Transformer layers
        num_heads = params['model_params']['Transformer']['num_heads']  # Number of attention heads
        dropout = params['dropout']  # Dropout rate for regularization

        # Transformer layer
        self.transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=hidden_dim, dropout=dropout)
        # Fully connected output layer
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        x = x.permute(1, 0, 2)  # Change shape to (sequence_length, batch_size, input_dim)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)  # Change back to (batch_size, sequence_length, input_dim)
        x = x[:, -1, :]  # Take the last time step
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class BiLSTM(NeuralNetworkModelBase):
    """Bidirectional LSTM model for sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including the number of features, 
            look-back period, prediction steps, and LSTM-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the BiLSTM model with the provided parameters."""
        super(BiLSTM, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        dropout = params['dropout']  # Dropout rate for regularization
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size=self.features_num, hidden_size=self.features_num, num_layers=self.num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        # Fully connected output layer
        self.fc = nn.Linear(self.features_num * 2, self.predict_steps)  # Multiply by 2 for bidirectional

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.fc(out)
        return out


class AttentionBiLSTM(NeuralNetworkModelBase):
    """Bidirectional LSTM model with attention mechanism for sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including the number of features, 
            look-back period, prediction steps, and LSTM-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the AttentionBiLSTM model with the provided parameters."""
        super(AttentionBiLSTM, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        dropout = params['dropout']  # Dropout rate for regularization

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size=self.features_num, hidden_size=self.features_num, num_layers=self.num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        # Attention mechanism layer
        self.attention = nn.Linear(self.features_num * 2, 1)
        # Fully connected output layer
        self.fc = nn.Linear(self.features_num * 2, self.predict_steps)  # Multiply by 2 for bidirectional

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(out), dim=1)
        out = torch.sum(out * attention_weights, dim=1)
        out = self.fc(out)
        return out


class CNN_LSTM(NeuralNetworkModelBase):
    """CNN-LSTM hybrid model for sequence prediction.

    This model first applies convolutional layers to extract features, 
    followed by an LSTM layer for sequence learning.

    Args:
        params (dict): Dictionary of model parameters including the number of features, 
            look-back period, prediction steps, and LSTM-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the CNN_LSTM model with the provided parameters."""
        super(CNN_LSTM, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        dropout = params['dropout']  # Dropout rate for regularization

        # Convolutional layer with group convolution (depthwise convolution)
        self.conv1 = nn.Conv1d(in_channels=self.features_num, out_channels=self.features_num, 
                               kernel_size=3, padding=1, groups=self.features_num)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        # LSTM layer with the specified number of features, layers, and dropout
        self.lstm = nn.LSTM(input_size=self.features_num, hidden_size=self.features_num, num_layers=self.num_layers, 
                            batch_first=True, dropout=dropout)
        # Fully connected output layer
        self.fc = nn.Linear(self.features_num, self.predict_steps)

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length) for Conv1D
        x = self.pool(F.relu(self.conv1(x)))  # Apply convolution, ReLU activation, and pooling
        x = self.dropout(x)  # Apply dropout
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, sequence_length, channels)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.fc(out)
        return out

class Chomp1d(NeuralNetworkModelBase):
    """Chomps off the padding added by the temporal convolution.

    This module is used to ensure that the output length matches the input length 
    by removing extra padding.

    Args:
        chomp_size (int): The size of padding to remove.
    """
    
    def __init__(self, chomp_size):
        """Initializes the Chomp1d module with the provided chomp size."""
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """Defines the forward pass of the Chomp1d module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with chomped dimensions.
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(NeuralNetworkModelBase):
    """A single temporal block for the Temporal Convolutional Network.

    Each temporal block consists of two convolutional layers with dilations, 
    followed by ReLU activations and dropout, and a residual connection.

    Args:
        n_inputs (int): Number of input channels.
        n_outputs (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolution.
        dilation (int): Dilation factor for the convolution.
        padding (int): Padding added to the input.
        dropout (float): Dropout rate for regularization.
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """Initializes the TemporalBlock with the provided parameters."""
        super(TemporalBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # Remove extra padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)  # Remove extra padding
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Define the network with a residual connection
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the convolutional layers."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """Defines the forward pass of the TemporalBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the temporal block.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(NeuralNetworkModelBase):
    """Temporal Convolutional Network for sequence prediction.

    This model applies a series of temporal blocks with increasing dilations to 
    capture long-range dependencies in the input sequence.

    Args:
        num_inputs (int): Number of input channels.
        num_channels (list): List of output channels for each temporal block.
        kernel_size (int): Size of the convolutional kernel.
        dropout (float): Dropout rate for regularization.
    """
    
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """Initializes the TemporalConvNet with the provided parameters."""
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Defines the forward pass of the TemporalConvNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the temporal convolutional network.
        """
        return self.network(x)

class TCN(NeuralNetworkModelBase):
    """Temporal Convolutional Network (TCN) model for sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and dropout rate.
    """
    
    def __init__(self, params):
        """Initializes the TCN model with the provided parameters."""
        super(TCN, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        dropout = params['dropout']  # Dropout rate for regularization
        input_size = self.look_back
        num_channels = [self.features_num] * 3
        kernel_size = 3
        output_size = self.predict_steps

        # Temporal Convolutional Network layer
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        # Fully connected output layer
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """Defines the forward pass of the TCN model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, features_num, sequence_length).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        y1 = self.tcn(x)
        o = self.linear(y1[:, :, -1])
        return o


class SelfAttention(NeuralNetworkModelBase):
    """Self-Attention model for sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and dropout rate.
    """
    
    def __init__(self, params):
        """Initializes the SelfAttention model with the provided parameters."""
        super(SelfAttention, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        dropout = params['dropout']  # Dropout rate for regularization
        embed_dim = self.features_num
        num_heads = 4

        # Multi-head Self-Attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # Fully connected output layer
        self.fc = nn.Linear(embed_dim, self.predict_steps)
    
    def forward(self, x):
        """Defines the forward pass of the SelfAttention model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        x = x.permute(1, 0, 2)  # Change shape to (sequence_length, batch_size, embed_dim)
        attn_output, _ = self.self_attention(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # Change back to (batch_size, sequence_length, embed_dim)
        out = self.dropout(attn_output[:, -1, :])  # Take the last time step and apply dropout
        out = self.fc(out)
        return out


class LSTM_many_to_many(NeuralNetworkModelBase):
    """LSTM model for many-to-many sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and LSTM-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the LSTM_many_to_many model with the provided parameters."""
        super(LSTM_many_to_many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        dropout = params['dropout']  # Dropout rate for regularization

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.features_num, hidden_size=self.features_num, 
                            num_layers=self.num_layers, batch_first=True, dropout=dropout)
        # Fully connected output layer
        self.fc = nn.Linear(self.look_back * self.features_num, self.predict_steps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Defines the forward pass of the LSTM_many_to_many model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.lstm(x)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class GRU_many_to_many(NeuralNetworkModelBase):
    """GRU model for many-to-many sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and GRU-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the GRU_many_to_many model with the provided parameters."""
        super(GRU_many_to_many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.num_layers = params['model_params']['GRU']['num_layers']  # Number of GRU layers
        dropout = params['dropout']  # Dropout rate for regularization

        # GRU layer
        self.gru = nn.GRU(input_size=self.features_num, hidden_size=self.features_num, 
                          num_layers=self.num_layers, batch_first=True, dropout=dropout)
        # Fully connected output layer
        self.fc = nn.Linear(self.look_back * self.features_num, self.predict_steps)  # Assuming input length is 64
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Defines the forward pass of the GRU_many_to_many model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.gru(x)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class AttentionLSTM_many_to_many(NeuralNetworkModelBase):
    """Attention-enhanced LSTM model for many-to-many sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and LSTM-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the AttentionLSTM_many_to_many model with the provided parameters."""
        super(AttentionLSTM_many_to_many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        dropout = params['dropout']  # Dropout rate for regularization

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.features_num, hidden_size=self.features_num, 
                            num_layers=self.num_layers, batch_first=True, dropout=dropout)
        # Attention mechanism layer
        self.attention = nn.Linear(self.features_num, 1)
        # Fully connected output layer
        self.fc = nn.Linear(self.look_back * self.features_num, self.predict_steps)  # Assuming input length is 64
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Defines the forward pass of the AttentionLSTM_many_to_many model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(out), dim=1)
        out = out * attention_weights
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class CNN_many_to_many(NeuralNetworkModelBase):
    """Convolutional Neural Network (CNN) model for many-to-many sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and CNN-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the CNN_many_to_many model with the provided parameters."""
        super(CNN_many_to_many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        dropout = params['dropout']  # Dropout rate for regularization

        # Convolutional layer with group convolution (depthwise convolution)
        self.conv1 = nn.Conv1d(in_channels=self.features_num, out_channels=self.features_num, 
                               kernel_size=3, padding=1, groups=self.features_num)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Fully connected output layer
        self.fc1 = nn.Linear(self.features_num * int(self.look_back / 2), self.predict_steps)  # Assuming input length is 64 and max pooling reduces it by half
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Defines the forward pass of the CNN_many_to_many model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class TransformerModel_many_to_many(NeuralNetworkModelBase):
    """Transformer model for many-to-many sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and Transformer-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the TransformerModel_many_to_many with the provided parameters."""
        super(TransformerModel_many_to_many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        input_dim = self.features_num
        hidden_dim = self.look_back 
        output_dim = self.predict_steps 
        num_layers = params['model_params']['Transformer']['num_layers']  # Number of Transformer layers
        num_heads = params['model_params']['Transformer']['num_heads']  # Number of attention heads
        dropout = params['dropout']  # Dropout rate for regularization

        # Transformer layer
        self.transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, 
                                          num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
                                          dim_feedforward=hidden_dim, dropout=dropout)
        # Fully connected output layer
        self.fc_out = nn.Linear(self.features_num * self.look_back, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Defines the forward pass of the TransformerModel_many_to_many.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        x = x.permute(1, 0, 2)  # Change shape to (sequence_length, batch_size, input_dim)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)  # Change back to (batch_size, sequence_length, input_dim)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class BiLSTM_many_to_many(NeuralNetworkModelBase):
    """Bidirectional LSTM (BiLSTM) model for many-to-many sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and LSTM-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the BiLSTM_many_to_many model with the provided parameters."""
        super(BiLSTM_many_to_many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        dropout = params['dropout']  # Dropout rate for regularization

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size=self.features_num, hidden_size=self.features_num, 
                            num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        # Fully connected output layer
        self.fc = nn.Linear(self.look_back * 2 * self.features_num, self.predict_steps)  # Assuming input length is 64 and bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Defines the forward pass of the BiLSTM_many_to_many model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.lstm(x)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class AttentionBiLSTM_many_to_many(NeuralNetworkModelBase):
    """Attention-enhanced Bidirectional LSTM (BiLSTM) model for many-to-many sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and LSTM-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the AttentionBiLSTM_many_to_many model with the provided parameters."""
        super(AttentionBiLSTM_many_to_many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        dropout = params['dropout']  # Dropout rate for regularization

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size=self.features_num, hidden_size=self.features_num, 
                            num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        # Attention mechanism layer
        self.attention = nn.Linear(self.features_num * 2, 1)
        # Fully connected output layer
        self.fc = nn.Linear(self.look_back * 2 * self.features_num, self.predict_steps)  # Assuming input length is 64 and bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Defines the forward pass of the AttentionBiLSTM_many_to_many model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(out), dim=1)
        out = out * attention_weights
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class CNN_LSTM_many_to_many(NeuralNetworkModelBase):
    """CNN-LSTM hybrid model for many-to-many sequence prediction.

    This model first applies convolutional layers to extract features, 
    followed by an LSTM layer for sequence learning.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and LSTM-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the CNN_LSTM_many_to_many model with the provided parameters."""
        super(CNN_LSTM_many_to_many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        self.num_layers = params['model_params']['LSTM']['num_layers']  # Number of LSTM layers
        dropout = params['dropout']  # Dropout rate for regularization

        # Convolutional layer with group convolution (depthwise convolution)
        self.conv1 = nn.Conv1d(in_channels=self.features_num, out_channels=self.features_num, 
                               kernel_size=3, padding=1, groups=self.features_num)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.features_num, hidden_size=self.features_num, 
                            num_layers=self.num_layers, batch_first=True, dropout=dropout)
        # Fully connected output layer
        self.fc = nn.Linear(self.features_num * int(self.look_back / 2), self.predict_steps)  # Assuming max pooling reduces length to 32
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Defines the forward pass of the CNN_LSTM_many_to_many model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, sequence_length, channels)
        out, _ = self.lstm(x)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class TCN_many_to_many(NeuralNetworkModelBase):
    """Temporal Convolutional Network (TCN) model for many-to-many sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and TCN-specific settings.
    """
    
    def __init__(self, params):
        """Initializes the TCN_many_to_many model with the provided parameters."""
        super(TCN_many_to_many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        dropout = params['dropout']  # Dropout rate for regularization

        # Temporal Convolutional Network layer
        num_channels = [self.features_num] * 3
        self.tcn = TemporalConvNet(num_inputs=self.features_num, num_channels=num_channels, dropout=dropout)
        # Fully connected output layer
        self.fc = nn.Linear(int(self.features_num * self.look_back), self.predict_steps)  # Assuming input length is 64
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Defines the forward pass of the TCN_many_to_many model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        out = self.tcn(x.permute(0, 2, 1))
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class SelfAttention_many_to_many(NeuralNetworkModelBase):
    """Self-Attention model for many-to-many sequence prediction.

    Args:
        params (dict): Dictionary of model parameters including feature columns, 
            look-back period, prediction steps, and self-attention settings.
    """
    
    def __init__(self, params):
        """Initializes the SelfAttention_many_to_many model with the provided parameters."""
        super(SelfAttention_many_to_many, self).__init__()
        self.params = params
        self.features_num = len(params['feature_cols'])  # Number of features in the input data
        self.look_back = params['look_back']  # Number of past time steps used as input
        self.predict_steps = params['predict_steps']  # Number of future steps to predict
        dropout = params['dropout']  # Dropout rate for regularization

        # Self-attention mechanism
        self.hidden_dim = self.features_num
        self.query = nn.Linear(self.features_num, self.look_back)
        self.key = nn.Linear(self.features_num, self.look_back)
        self.value = nn.Linear(self.features_num, self.look_back)
        # Fully connected output layer
        self.fc = nn.Linear(self.look_back * self.look_back, self.predict_steps)  # Assuming input length is 64
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Defines the forward pass of the SelfAttention_many_to_many model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, features_num).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, predict_steps).
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5), dim=-1)
        out = torch.matmul(attention_weights, V)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class ModelFactory:
    @staticmethod
    def create_model_instance(model_type, params):
        """
        Creates an instance of the specified model type.

        Args:
            model_type (str): The type of the model to create. This should match one of the keys in the `models` dictionary.
            params (dict): A dictionary containing the parameters for the model. These parameters are passed to the model's constructor.

        Returns:
            instance: An instance of the specified model type.

        Raises:
            ValueError: If the provided `model_type` is not recognized.
        """
        
        # Dictionary mapping model type strings to the actual model classes
        models = {
            'LSTM': LSTM,
            'LSTM_many_to_many': LSTM_many_to_many,
            'AttentionLSTM': AttentionLSTM,
            'AttentionLSTM_many_to_many': AttentionLSTM_many_to_many,
            'GRU': GRU,
            'GRU_many_to_many': GRU_many_to_many,
            'CNN': CNN,
            'CNN_many_to_many': CNN_many_to_many,
            'TransformerModel': TransformerModel,
            'TransformerModel_many_to_many': TransformerModel_many_to_many,
            'BiLSTM': BiLSTM,
            'BiLSTM_many_to_many': BiLSTM_many_to_many,
            'AttentionBiLSTM': AttentionBiLSTM,
            'AttentionBiLSTM_many_to_many': AttentionBiLSTM_many_to_many,
            'CNN_LSTM': CNN_LSTM,
            'CNN_LSTM_many_to_many': CNN_LSTM_many_to_many,
            'TCN': TCN,
            'TCN_many_to_many': TCN_many_to_many,
            'SelfAttention': SelfAttention,
            'SelfAttention_many_to_many': SelfAttention_many_to_many,
            # Add other models here as needed
        }
        
        # Retrieve the model class from the dictionary
        model_instance = models.get(model_type)
        
        # If the model type is not found, raise an error
        if model_instance is None:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # Create an instance of the model with the provided parameters
        instance = model_instance(params)
        
        return instance
