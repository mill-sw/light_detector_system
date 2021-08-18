import tensorflow.keras as k

# def build_model(hp):
#   model = k.Sequential()
#   model.add(k.layers.Dense(hp.Choice('units', [8, 16, 32]), activation='relu'))
#   model.add(k.layers.Dense(1, activation='sigmoid'))
#   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#   return model



def model_dense(inn, out):
    input = k.Input(shape=(inn, ))

    x = k.layers.Dense(16, activation="relu")(input)
    x = k.layers.Dense(14, activation="relu")(x)
    x = k.layers.Dense(12, activation="relu")(x)
    x = k.layers.Dense(10, activation="relu")(x)
    x = k.layers.Dense(8, activation="relu")(x)
    x = k.layers.Dense(6, activation="relu")(x)

    output = k.layers.Dense(out, activation="softmax")(x)

    model = k.Model(input, output)

    return model


def model_lstm(inn, out):
    input = k.Input(shape=(None, inn))

    x = k.layers.LSTM(8, activation='tanh', recurrent_activation='sigmoid', dropout=0.4, recurrent_dropout=0, unroll=False, use_bias=True)(input)
    # x = k.layers.Dense(64, activation="relu")(x)
    # x = k.layers.Dense(32, activation="relu")(x)
    # x = k.layers.Dense(16, activation="relu")(x)
    x = k.layers.Dense(4, activation="relu")(x)

    output = k.layers.Dense(out, activation="softmax")(x)

    model = k.Model(input, output)

    return model