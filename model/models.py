import tensorflow.keras as k

# def build_model(hp):
#   model = k.Sequential()
#   model.add(k.layers.Dense(hp.Choice('units', [8, 16, 32]), activation='relu'))
#   model.add(k.layers.Dense(1, activation='sigmoid'))
#   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#   return model

# tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10)
# tuner.search(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
# best_model = tuner.get_best_models()[0]

def model_dense(inn, out):
    input = k.Input(shape=(inn, ))
    # input = k.Input(shape=(None, inn))

    # x = k.layers.LSTM(12, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0, unroll=False, use_bias=True)(input)
    x = k.layers.Dense(16, activation="relu")(input)
    # x = k.layers.Dense(64, activation="relu")(x)
    # x = k.layers.Dense(32, activation="relu")(x)
    # x = k.layers.Dense(16, activation="relu")(x)
    x = k.layers.Dense(8, activation="relu")(x)

    output = k.layers.Dense(out, activation="softmax")(x)

    model = k.Model(input, output)

    return model

def model_lstm(inn, out):
    input = k.Input(shape=(inn, ))
    # input = k.Input(shape=(None, inn))

    x = k.layers.LSTM(12, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0, unroll=False, use_bias=True)(input)
    # x = k.layers.Dense(16, activation="relu")(input)
    # x = k.layers.Dense(64, activation="relu")(x)
    # x = k.layers.Dense(32, activation="relu")(x)
    # x = k.layers.Dense(16, activation="relu")(x)
    x = k.layers.Dense(8, activation="relu")(x)

    output = k.layers.Dense(out, activation="softmax")(x)

    model = k.Model(input, output)

    return model