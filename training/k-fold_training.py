import glob
import os

import pandas as pd
import keras_tuner as kt
import tensorflow.keras as k
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import EarlyStopping

from helper.path import *
from model.models import *
from postprocess.analyze import draw_CM, draw_ROC_AUC
from preprocess.preprocess import *

# # load tensorboard
# %load_ext tensorboard

classes = ["idle", "on", "off"]
class_list = list(range(len(classes)))

idle = 0
on = 1
off = 2

window = 6
input = 1
output = 3

num_folds = 5

# load data
path = '/home/z/data/label/'
for file in glob.glob(path + "*.csv"):
    print(" "*8, f"file loaded : {file}")
    print('-'*128)
    df = pd.read_csv(file)
    df = df[['device_field03', 'illuminance_onoff', 'device_data_reg_dtm']]
    light_data = df['device_field03']
    label_data = df['illuminance_onoff']
    time_data = df['device_data_reg_dtm']

    new_label = change_labels(label_data, on, off, idle)
    df.drop(columns='illuminance_onoff')
    df['illuminance_onoff'] = new_label

# preprocess
    fold_num = 1
    kfold = KFold(n_splits=num_folds, shuffle=False)

    norm_data = normalize(light_data)

    x1, y1 = to_sequences(window, norm_data, label_data)

    x, y = balance_data(x1, y1)

# load model
    try:
        model_list = glob.glob(f"{model_path}{date}/*.h5")
        model_name = max(model_list, key=os.path.getctime)
        model = k.models.load_model(model_name)
        basename = os.path.basename(model_name)
        print('\n'*4, f"retraining : {basename}", '\n'*4)
    except:
        model = model_dense(window, output)
        # model = model_lstm(input, output)
        print('\n'*4, f"training new model", '\n'*4)

    # compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

# fit model
    for train, test in kfold.split(x, y):
        print(f'  fold {fold_num}                {file}')
        print('-'*128)

        tensorboard_callback = k.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
        es = EarlyStopping(monitor="val_loss", patience=8, mode="auto", verbose=2)

# # keras tuner
#         tuner = kt.RandomSearch(model, objective='val_loss', max_trials=10)
#         tuner.search(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
#         best_model = tuner.get_best_models()[0]

        history = model.fit(x[train], y[train], validation_split=0.1, batch_size=8,
                            epochs=1024, verbose=2, callbacks=[es, tensorboard_callback])
        # plot
        pd.DataFrame(history.history).plot(figsize=(16, 10), grid=1, xlabel="epoch", ylabel="accuracy")
        plt.show()

        print("\n", "evaluate : ")
        loss, acc = model.evaluate(x[test], y[test], verbose=1)
        fold_num += 1
        print('-'*128, '\n'*4)


# save model
    new_model_name = model_path + f"{date}/" + f"light_detector_{time}" + ".h5"
    model.save(new_model_name)

# # predict
#     predict = model.predict(x[test])
#     predicted = np.argmax(predict, axis=1)
#
# # CM
#     draw_CM(y[test], predicted)
#
# # ROC, AUC
#     x = label_binarize(predicted, classes=class_list)
#     y = label_binarize(y[test], classes=class_list)
#
#     draw_ROC_AUC(x, y, class_list)

# # launch tensorboard @ localhost:6006
    # %tensorboard --logdir logs/ --host localhost --port 6006
