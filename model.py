import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint



class LSTM_model:

    def __init__(self, training_features, training_labels, validation_features, validation_labels):

        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(training_features.shape[1:]), return_sequences=True, activation="tanh"))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())

        self.model.add(LSTM(128, input_shape=(training_features.shape[1:]), return_sequences=True, activation="tanh"))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())

        self.model.add(LSTM(128, input_shape=(training_features.shape[1:]), activation="tanh"))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())

        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(2, activation="softmax"))

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=0.000001)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])

        self.tensorboard = TensorBoard(log_dir="log/m_{}".format(time.time))

        self.LSTM_rnn = self.model.fit(training_features, training_labels, batch_size=30, epochs=8, validation_data=(validation_features, validation_labels), callbacks=[self.tensorboard])
