import os

import pandas as pd
import plotly.express as px

idle = 0
on = 1
off = 2

# file = '/home/z/data/label/labeling_NW0513.csv'
# file = '/home/z/data/label/labeling_NW0479.csv'
file = '/home/z/data/label/labeling_NW0520.csv'

df = pd.read_csv(file)
basename = os.path.basename(file)
df = df[['device_field03', 'illuminance_onoff', 'device_data_reg_dtm']]
light_data = df['device_field03']
label_data = df['illuminance_onoff']
time_data = df['device_data_reg_dtm']

df = df.assign(average=light_data)

# relabel to 0, 1, 2 = idel, on, off
prev = 0
for row, label in label_data.iteritems():
    if prev == 0 and label == 1:
        df.loc[row, 'illuminance_onoff'] = on
    elif prev == 1 and label == 0:
        df.loc[row, 'illuminance_onoff'] = off
    else:
        df.loc[row, 'illuminance_onoff'] = idle
    prev = label

# make average column
av = 0
ave = 0
for row, now in light_data.iteritems():
    data = (now + ave) / 2
    ave = data
    df.at[row, 'average'] = data
average = df['average']

# plot
fig = px.line(df, x='device_data_reg_dtm', y=['device_field03', 'illuminance_onoff', 'average'])
# fig = px.line(df, x='device_data_reg_dtm', y=['device_field03', 'average_filter', 'average', 'illuminance_onoff'])
fig.update_xaxes(rangeslider_visible=True)
fig.show(renderer='browser')

