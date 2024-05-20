import numpy as np
# import tensorflow as tf
import random
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation
# from keras.optimizers import Adam
# from keras import regularizers
# from keras import metrics
# from keras.callbacks import EarlyStopping
# from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm
import datetime
# from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


class ModelBase:
    def __init__(self, model=None, params=None, input_shape=None):
        self.model = model
        self.params = params
        self.input_shape = input_shape

    def create(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def train(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def infer(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def online_train(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate(self, y_preds, y_test):
        # Flatten the 3D tensors for evaluation
        y_test_flat = np.argmax(y_test.reshape(-1, y_test.shape[-1]), axis=1)
        y_preds_flat = np.argmax(
            y_preds.reshape(-1, y_preds.shape[-1]), axis=1)

        # Calculate evaluation metrics
        precision = precision_score(y_test_flat, y_preds_flat, average='macro')
        recall = recall_score(y_test_flat, y_preds_flat, average='macro')
        accuracy = accuracy_score(y_test_flat, y_preds_flat)
        f1 = f1_score(y_test_flat, y_preds_flat, average='macro')

        return precision, recall, accuracy, f1


# class NeuralNetworkModelBase(ModelBase):
#     def train(self, X_train, y_train):
#         # Define callbacks, such as EarlyStopping
#         early_stopping = EarlyStopping(
#             monitor='loss',
#             patience=self.params.get('patience', 5),
#             restore_best_weights=True
#         )
#         # Train the model
#         history = self.model.fit(
#             X_train, y_train,
#             epochs=self.params.get('epochs', 10),
#             batch_size=self.params.get('batch_size', 32),
#             callbacks=[early_stopping],
#             verbose=2
#         )
#         return history

#     def infer(self, X_test):
#         # Predict the next instance (optional, depending on your requirements)
#         y_pred = self.model.predict(X_test)
#         return y_pred


class NeuralNetworkModelBase(ModelBase, nn.Module):
    def __init__(self):
        super(NeuralNetworkModelBase, self).__init__()

    def train(self, X_train, y_train, params):
        self.train()
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=params.get('batch_size', 32))

        optimizer = Adam(self.parameters(), lr=params.get('learning_rate', 0.001))
        criterion = nn.BCELoss()

        for epoch in range(params.get('epochs', 10)):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

    def infer(self, X_test):
        self.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_pred = self(X_test_tensor)
        return y_pred.numpy()
    
    def online_train(self, X_train, y_train, single_X_test, single_y_test, data_update_mode='newest'):
        # Update the training dataset with the new instance
        if data_update_mode == 'append':
            online_X_train = np.append(X_train, single_X_test, axis=0)
            online_y_train = np.append(y_train, single_y_test, axis=0)
        elif data_update_mode == 'replace':
            online_X_train = np.append(X_train[1:], single_X_test, axis=0)
            online_y_train = np.append(y_train[1:], single_y_test, axis=0)
        elif data_update_mode == 'newest':
            online_X_train = single_X_test
            online_y_train = single_y_test
        else:
            raise ValueError(f"Invalid data update mode: {data_update_mode}")

        # Add the instance and its actual result to the training dataset
        X_train = np.append(X_train, single_X_test, axis=0)
        y_train = np.append(y_train, single_y_test, axis=0)

        # Retrain the model on this updated dataset
        history = self.model.fit(
            online_X_train, online_y_train,
            epochs=1, verbose=2
        )
        return history


class Modelseq2seq(nn.Module):
    def __init__(self, params, input_shape):
        super(Modelseq2seq, self).__init__()
        # Assuming input_shape is a tuple: (channels, height, width)
        channels, height, width = input_shape

        # Example of Convolutional and LSTM layers based on your original structure
        self.conv1 = nn.Conv1d(channels, params['conv_1_filter'], params['conv_1_kernel'], padding='same')
        self.bn1 = nn.BatchNorm1d(params['conv_1_filter'])
        self.dropout1 = nn.Dropout(params['dropout_1'])
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(params['conv_1_filter'], params['conv_2_filter'], params['conv_2_kernel'])
        self.bn2 = nn.BatchNorm1d(params['conv_2_filter'])
        self.dropout2 = nn.Dropout(params['dropout_2'])
        self.pool2 = nn.MaxPool1d(2)

        self.lstm1 = nn.LSTM(params['conv_2_filter'] * (width // 4), params['lstm_1_units'], batch_first=True)

        # Decoder part
        self.repeat = nn.Linear(params['lstm_1_units'], params['predict_steps'] * params['lstm_1_units'])
        self.lstm2 = nn.LSTM(params['lstm_1_units'], params['lstm_2_units'], batch_first=True)
        self.output_layer = nn.Linear(params['lstm_2_units'], 2)

    def forward(self, x, params):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        # Reshaping for LSTM input
        x = x.view(x.size(0), -1)
        x, _ = self.lstm1(x.unsqueeze(1))

        # Repeat vector
        x = self.repeat(x).view(x.size(0), params['predict_steps'], -1)

        # Decoder
        x, _ = self.lstm2(x)
        x = self.output_layer(x)

        return x

# class Modelseq2seq(NeuralNetworkModelBase):
#     def create(self):
#         model = Sequential()
#         print(self.params)
#         # Encoder
#         model.add(Conv1D(filters=self.params['conv_1_filter'],
#                          kernel_size=self.params['conv_1_kernel'],
#                          activation=None,
#                          padding='same',
#                          kernel_regularizer=regularizers.l2(
#                              self.params['conv_1_l2']),
#                          input_shape=self.input_shape))
#         model.add(BatchNormalization())
#         model.add(Activation('relu'))
#         model.add(Dropout(self.params['dropout_1']))
#         model.add(MaxPooling1D(pool_size=2))

#         model.add(Conv1D(filters=self.params['conv_2_filter'],
#                          kernel_size=self.params['conv_2_kernel'],
#                          activation=None))
#         model.add(BatchNormalization())
#         model.add(Activation('relu'))
#         model.add(Dropout(self.params['dropout_2']))
#         model.add(MaxPooling1D(pool_size=2))

#         model.add(LSTM(units=self.params['lstm_1_units'],
#                        activation=None,
#                        return_sequences=False,
#                        kernel_regularizer=regularizers.l2(self.params['lstm_1_l2'])))
#         model.add(BatchNormalization())
#         model.add(Activation('tanh'))
#         model.add(Dropout(self.params['dropout_3']))

#         # Set the desired output sequence length using RepeatVector
#         model.add(RepeatVector(self.params['predict_steps']))

#         # Decoder
#         model.add(LSTM(units=self.params['lstm_2_units'],
#                        activation=None,
#                        return_sequences=True))
#         model.add(BatchNormalization())
#         model.add(Activation('tanh'))
#         model.add(Dropout(self.params['dropout_4']))
#         model.add(TimeDistributed(Dense(2, activation='softmax')))

#         optimizer = Adam(learning_rate=self.params['learning_rate'])
#         model.compile(optimizer=optimizer,
#                       loss='binary_crossentropy',
#                       metrics=[metrics.BinaryAccuracy()])
#         model.summary()
#         self.model = model


# class ModelCNN(NeuralNetworkModelBase):
#     pass

# class ModelFFT(ModelBase):
#     pass

# class ModelMA(ModelBase):
#     pass


class ModelFactory:
    @staticmethod
    def create_model_instance(model_type, params=None, input_shape=None, keras_model=None, *args, **kwargs):
        models = {
            "seq2seq": Modelseq2seq,
            # "cnn": ModelCNN,
            # "fft": ModelFFT,
            # "ma": ModelMA,
            # Add other models here as needed
        }
        model_instance = models.get(model_type)
        if model_instance is None:
            raise ValueError(f"Invalid model type: {model_type}")
        if keras_model:
            instance = model_instance(keras_model)
        else:
            instance = model_instance(None, params, input_shape)
        return instance


# class Model:
#     def create_model(self, model_type, params=None, input_shape=None):
#         # Create the model instance
#         model_instance = ModelFactory.create_model_instance(
#             model_type, params, input_shape)
#         # Ensure the model is created (initialized)
#         model_instance.create()
#         return model_instance

#     def train_model(self, model, X_train, y_train):
#         # Train the model
#         return model.train(X_train, y_train)

#     def infer_model(self, model, X_test):
#         # Perform inference using the model
#         return model.infer(X_test)

#     def online_train_model(self, model, X_train, y_train, single_X_test, single_y_test):
#         # Perform online training on the model
#         return model.online_train(X_train, y_train, single_X_test, single_y_test)

#     def evaluate_model(self, model, y_preds, y_test):
#         # Evaluate the model
#         return model.evaluate(y_preds, y_test)

#     def save_model(self, model, base_filename='model'):
#         # Create a timestamp or unique identifier
#         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         filename = f'{base_filename}_{timestamp}.h5'

#         # Save the model
#         model.model.save(filename)
#         print(f'Model saved as {filename}')

#     def load_model(self, model_type, model_path):
#         # Load the pre-trained Keras model
#         loaded_keras_model = load_model(model_path)
#         print(f'Model loaded from {model_path}')

#         # Wrap the Keras model in the appropriate custom model class
#         model_instance = ModelFactory.create_model_instance(
#             model_type, keras_model=loaded_keras_model)
#         return model_instance


class Model:
    def create_model(self, model_type, params=None, input_shape=None):
        model_instance = ModelFactory.create_model_instance(model_type, params, input_shape)
        return model_instance

    def train_model(self, model, X_train, y_train, params):
        model.train_model(X_train, y_train, params)

    def infer_model(self, model, X_test):
        return model.infer(X_test)

    def save_model(self, model, file_path):
        torch.save(model.state_dict(), file_path)

    def load_model(self, model, file_path):
        model.load_state_dict(torch.load(file_path))
        model.eval()
        return model
    
    def online_train_model(self, model, X_train, y_train, single_X_test, single_y_test):
        # Perform online training on the model
        return model.online_train(X_train, y_train, single_X_test, single_y_test)

    def evaluate_model(self, model, y_preds, y_test):
        # Evaluate the model
        return model.evaluate(y_preds, y_test)

    def run(self, model_type, look_back, params, X_train, y_train, X_test, y_test, pre_trained_model_path=None):
        if pre_trained_model_path:
            # Load the pre-trained model
            model = self.load_model(model_type, pre_trained_model_path)
            history = None
        else:
            # Create a new model if no pre-trained model is provided
            input_shape = (look_back, X_train.shape[-1])
            model = self.create_model(model_type, params, input_shape)
            # Train the new model
            history = self.train_model(model, X_train, y_train)
            # Save the initial trained model
            self.save_model(model)

        online_training_losses = []
        online_training_acc = []
        y_preds = []
        for i in tqdm(range(len(X_test))):
            # Predict the next instance
            y_pred = self.infer_model(model, X_test[i:i+1])
            y_preds.append(y_pred[0])

            # Perform online training
            online_history = self.online_train_model(
                model, X_train, y_train, X_test[i:i+1], y_test[i:i+1])
            online_training_losses.append(online_history.history['loss'][0])
            online_training_acc.append(
                online_history.history['binary_accuracy'][0])
        y_preds = np.array(y_preds)

        # Save the model after each online training iteration
        self.save_model(model)

        return model, history, y_preds, online_training_losses, online_training_acc
