import glob
import os
from datetime import datetime as dt

import pandas as pd
import keras_tuner as kt
import tensorflow.keras as k
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px

from model.models import *
from preprocess.preprocess import *


light_col = "조도"
date_col = "측정일시 koKR"

idle = 0
on = 1
off = 2

window = 6

date = dt.now().strftime("%Y%m%d")
time = dt.now().strftime("%H%M%S")

model_path = "trained_models/"

path = '/home/z/data/test1/'
for file in glob.glob(path + "*.csv"):
    print(f"file loaded : {file}")
    df = pd.read_csv(file)
    if len(df.index) < 100: continue

    df = df[[light_col, date_col]]
    light_data = df[light_col]
    time_data = df[date_col]

# preprocess
    norm_data = normalize(light_data)

    x = inference_sequences(window, norm_data)

# load
    model_list = glob.glob(f"{model_path}{date}/*.h5")
    model_name = max(model_list, key=os.path.getctime)
    model = k.models.load_model(model_name)
    basename = os.path.basename(model_name)
    print(f"model loaded : {basename}")

# predict
    predict = model.predict(x)
    predicted = np.argmax(predict, axis=1)

    paddings = [0]*(window-1)
    pred = predicted.tolist()
    inference = paddings + pred +[0]
    df['predicted'] = inference

# # plot
#     fig = px.line(df, x=date_col, y=[light_col, 'predicted'])
#     fig.update_xaxes(rangeslider_visible=True)
#     fig.show(renderer='browser')

# save to csv
    filename = os.path.basename(file)
    df.to_csv(f"result/{filename}", index=False)
