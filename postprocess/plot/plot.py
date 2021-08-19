import glob
import os

import pandas as pd
import plotly.express as px

from postprocess.analyze import draw_CM

idle = 0
on = 1
off = 2

path = '/home/z/data/label/'
for file in glob.glob("*.csv"):
    df = pd.read_csv(file)
    basename = os.path.basename(file)
    df = df[['device_field03', 'illuminance_onoff', 'device_data_reg_dtm']]
    light_data = df['device_field03']
    label_data = df['illuminance_onoff']
    time_data = df['device_data_reg_dtm']

    light_data = df["light"]

    df = df.assign(average=light_data)
    df = df.assign(average_filter=light_data)
    average_filter = df['average_filter']


# make average column
    av = 0
    ave = 0
    for row, now in light_data.iteritems():
        data = (now + ave) / 2
        ave = data
        df.at[row, 'average'] = data
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
            # df.loc[row, 'average_filter'] = idle
            continue
        if (r - r1) > (n - n1) + offset and (r - r1) - (n - n1) > threshold:
            av.append(on)
            # df.loc[row, 'average_filter'] = on
        elif (r1 - r) > (n1 - n) + offset and (r1 - r) - (n1 - n) > threshold:
            av.append(off)
            # df.loc[row, 'average_filter'] = off
        else:
            av.append(idle)
            # df.loc[row, 'average_filter'] = idle
    df["average_filter"] = av
    average_filter = df['average_filter']


# plot
    fig = px.line(df, x='device_data_reg_dtm', y=['device_field03', 'average_filter', 'average', 'illuminance_onoff'])
    fig.update_xaxes(rangeslider_visible=True)
    fig.show(renderer='browser')

# cm
    draw_CM(df['illuminance_onoff'], average_filter)