
import glob
import os
from datetime import datetime as dt

import pandas as pd
import tensorflow.keras as k
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping

from model.models import model_dense
from preprocess.preprocess import *

# # load tensorboard
# %load_ext tensorboard

start = 230
end = 245
classes = ["idle", "on", "off"]
class_list = list(range(len(classes)))
idle = 0
on = 1
off = 2
ratio = 0.9
window = 6
input = 1
output = 3
num_folds = 8

path = '/home/z/data/label/'
for file in glob.glob(path + "*.csv"):
    df = pd.read_csv(file)
    df = df[['device_field03', 'illuminance_onoff', 'device_data_reg_dtm']]
    light_data = df['device_field03']
    label_data = df['illuminance_onoff']
    time_data = df['device_data_reg_dtm']

    new_label = change_labels(label_data, on, off, idle)
    df.drop(columns='illuminance_onoff')
    df['illuminance_onoff'] = new_label

# preprocess
    norm_data = normalize(light_data)

    train_data, test_data = split(norm_data, ratio)
    train_label, test_label = split(label_data, ratio)

    x_train, y_train = to_sequences(window, train_data, train_label)
    x_test, y_test = to_sequences(window, test_data, test_label)

    x_train, y_train = balance_data(x_train, y_train)
    x_test, y_test = balance_data(x_test, y_test)

    # y_train = one_hot_encoding(y_train)
    # y_test = one_hot_encoding(y_test)


# load
    try:
        model_list = glob.glob('trained_models/*.h5')
        model_name = max(model_list, key=os.path.getctime)
        model = k.models.load_model(model_name)
    except:
        model = model_dense(window, output)
        # model = model_lstm(input, output)

# compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

# fit
    log_path = "logs/" + dt.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = k.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    es = EarlyStopping(monitor="val_loss", patience=10, mode="auto", verbose=2)

    history = model.fit(x_train, y_train, validation_split=0.1, batch_size=6,
                        epochs=1000, verbose=1, callbacks=[es])  # callbacks=[es, tensorboard_callback])

# # plot
#     pd.DataFrame(history.history).plot(figsize=(16, 10), grid=1, xlabel="epoch", ylabel="accuracy")
#     plt.show()
#
# # plot
#     fig = px.line(df, x='device_data_reg_dtm', y=['device_field03', 'illuminance_onoff'])
#     fig.update_xaxes(rangeslider_visible=True)
#     fig.show(renderer='browser')

# save
    model_name = "trained_model/light_detector_" + dt.now().strftime("%Y%m%d-%H%M%S") + ".h5"
    model.save(model_name)

# evauate
    new_model = k.models.load_model(model_name)
    loss, acc = new_model.evaluate(x_test, y_test, verbose=1)
    print(f'test_loss: {loss} test_accuracy: {acc}')

    predict = new_model.predict(x_test)
    predicted = np.argmax(predict, axis=1)

# # CM
#     draw_CM(y_test, predicted)
#
# # ROC, AUC
#     x = label_binarize(predicted, classes=class_list)
#     y = label_binarize(y_test, classes=class_list)
#
#     draw_ROC_AUC(x, y, class_list)

# launch tensorboard @ localhost:6006
    # %tensorboard --logdir logs/ --host localhost --port 6006

# save to csv
#     result = inference_to_df(new_model, df)
#     df['prediction'] = result
#     df.to_csv("data.csv", index=False)
