from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM


def keras_lstm(input_dim, output_dim):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(input_dim, 1),
                   return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(activation="linear", units=output_dim))

    model.compile(loss="mse", optimizer="rmsprop")

    return model
