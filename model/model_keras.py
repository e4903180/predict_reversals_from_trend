import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras import regularizers
from keras import metrics
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm
import datetime
from tensorflow.keras.models import load_model
from io import StringIO
import sys

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


class NeuralNetworkModelBase(ModelBase):
    def train(self, X_train, y_train):
        # Define callbacks, such as EarlyStopping
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=self.params.get('patience', 5),
            restore_best_weights=True
        )
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.params.get('epochs', 10),
            batch_size=self.params.get('batch_size', 32),
            callbacks=[early_stopping],
            verbose=2
        )
        return history

    def infer(self, X_test):
        # Predict the next instance (optional, depending on your requirements)
        y_pred = self.model.predict(X_test)
        return y_pred

    def online_train(self, X_train, y_train, single_X_test, 
                     single_y_test, data_update_mode='newest'):
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


class Modelseq2seq(NeuralNetworkModelBase):
    def create(self):
        model = Sequential()
        # Encoder
        model.add(Conv1D(filters=self.params['conv_1_filter'],
                         kernel_size=self.params['conv_1_kernel'],
                         activation=None,
                         padding='same',
                         kernel_regularizer=regularizers.l2(
                             self.params['conv_1_l2']),
                         input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.params['dropout_1']))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(filters=self.params['conv_2_filter'],
                         kernel_size=self.params['conv_2_kernel'],
                         activation=None))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.params['dropout_2']))
        model.add(MaxPooling1D(pool_size=2))

        model.add(LSTM(units=self.params['lstm_1_units'],
                       activation=None,
                       return_sequences=False,
                       kernel_regularizer=regularizers.l2(self.params['lstm_1_l2'])))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(self.params['dropout_3']))

        # Set the desired output sequence length using RepeatVector
        model.add(RepeatVector(self.params['predict_steps']))

        # Decoder
        model.add(LSTM(units=self.params['lstm_2_units'],
                       activation=None,
                       return_sequences=True))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(self.params['dropout_4']))
        model.add(TimeDistributed(Dense(2, activation='softmax')))

        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=[metrics.BinaryAccuracy()])
        model.summary()
        self.model = model


# class ModelCNN(NeuralNetworkModelBase):
#     pass

# class ModelFFT(ModelBase):
#     pass

# class ModelMA(ModelBase):
#     pass


class ModelFactory:
    @staticmethod
    def create_model_instance(model_type, params=None, 
                              input_shape=None, keras_model=None, *args, **kwargs):
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


class Model:
    def create_model(self, model_type, params=None, input_shape=None):
        # Create the model instance
        model_instance = ModelFactory.create_model_instance(
            model_type, params, input_shape)
        # Ensure the model is created (initialized)
        model_instance.create()
        return model_instance

    def train_model(self, model, X_train, y_train):
        # Train the model
        return model.train(X_train, y_train)

    def infer_model(self, model, X_test):
        # Perform inference using the model
        return model.infer(X_test)

    def online_train_model(self, model, X_train, 
                           y_train, single_X_test, single_y_test):
        # Perform online training on the model
        return model.online_train(X_train, y_train, single_X_test, single_y_test)

    def save_model(self, model, base_filename='model'):
        # Create a timestamp or unique identifier
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'model/saves/{base_filename}_{timestamp}.h5'

        # Save the model
        model.model.save(filename)
        print(f'Model saved as {filename}')

    def load_model(self, model_type, model_path):
        # Load the pre-trained Keras model
        loaded_keras_model = load_model(model_path)
        print(f'Model loaded from {model_path}')

        # Wrap the Keras model in the appropriate custom model class
        model_instance = ModelFactory.create_model_instance(
            model_type, keras_model=loaded_keras_model)
        return model_instance

    def run(self, model_type, look_back, params, X_train, 
            y_train, X_test, y_test, pre_trained_model_path=None):
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
