import glob
import os
from datetime import datetime as dt

import pandas as pd
import keras_tuner as kt
import tensorflow.keras as k
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping

from model.models import *
from preprocess.preprocess import *

idle = 0
on = 1
off = 2

window = 6

data = dt.now().strftime("%Y%m%d")
time = dt.now().strftime("%H%M%S")

path = '/home/z/data/test1/'
for file in glob.glob(path + "*.csv"):
    print('\n'*8)
    print(f"{file} loaded")
    df = pd.read_csv(file)
    df = df[['조', 'illuminance_onoff', 'device_data_reg_dtm']]
    light_data = df['device_field03']
    label_data = df['illuminance_onoff']
    time_data = df['device_data_reg_dtm']

    new_label = change_labels(label_data, on, off, idle)
    df.drop(columns='illuminance_onoff')
    df['illuminance_onoff'] = new_label

# preprocess

    norm_data = normalize(light_data)

    x, y = to_sequences(window, norm_data, label_data)

# load
    model_list = glob.glob('trained_model/*.h5')
    model_name = max(model_list, key=os.path.getctime)
    model = k.models.load_model(model_name)

# # predict
    predict = model.predict(x)
    predicted = np.argmax(predict, axis=1)

    paddings = [0]*window
    inference = paddings + predict

# # CM
    #     draw_CM(y_test, predicted)

# # ROC, AUC
    #     x = label_binarize(predicted, classes=class_list)
    #     y = label_binarize(y_test, classes=class_list)
    #
    #     draw_ROC_AUC(x, y, class_list)

# save to csv
    df['predicted'] = inference
    df.to_csv("data.csv", index=False)
