import glob, os

import pandas as pd
import plotly.express as px

from helper.path import csv_path

idle = 0
on = 1
off = 2

dtime = "측정일시 koKR"
light = "조도"
label = "predicted"

path = "/home/z/PycharmProjects/light_detector/result/csv/"
for file in glob.glob(path + "*.csv"):
    df = pd.read_csv(file)
    df = df[[dtime, light, label]]
    light_data = df[light]
    time_data = df[dtime]
    label_data = df[label]

# make average column
    ave = 0
    ave_list = []
    for row, now in light_data.iteritems():
        data = (now + ave) / 2
        d1 = float("{:.2f}".format(data))
        ave_list.append(d1)
    df["average"] = ave_list
    average = df['average']

# average filter
    r1 = 0
    n1 = 0
    min_val = 9
    offset = 2
    threshold = 7.5
    av = []
    for row, n in average.iteritems():
        r = light_data.loc[row]
        if row > 0:
            r1 = light_data.loc[row - 1]
            n1 = average.loc[row - 1]
        if abs(r - r1) < min_val:
            av.append(idle)
            continue
        if (r - r1) > (n - n1) + offset and (r - r1) - (n - n1) > threshold:
            av.append(on)
        elif (r1 - r) > (n1 - n) + offset and (r1 - r) - (n1 - n) > threshold:
            av.append(off)
        else:
            av.append(idle)
    df["average_filter"] = av
    average_filter = df['average_filter']

# # plot
#     fig = px.line(df, x=dtime, y=["조도", label_data, average_filter])
#     fig.update_xaxes(rangeslider_visible=True)
#     fig.show(renderer='browser')

# save to csv
    filename = os.path.basename(file)
    save_path = "/home/z/PycharmProjects/light_detector/result/filter/"
    df.to_csv(save_path + filename, index=False)
