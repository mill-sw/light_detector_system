import numpy as np


# 9 to 0
def change_labels(data, on, off, idle):
    prev = 0
    new_label = []
    for row, label in data.iteritems():
        if prev == 0 and label == 1:
            new_label.append(on)
            # df.at[row, 'illuminance_onoff'] = on
        elif prev == 1 and label == 0:
            new_label.append(off)
            # df.at[row, 'illuminance_onoff'] = off
        else:
            new_label.append(idle)
            # df.at[row, 'illuminance_onoff'] = idle
        prev = label

    return new_label


def normalize(data):
    mx = data.max()
    norm_data = data / mx

    return norm_data


def split(data, ratio):
    split_index = int(len(data) * ratio)
    d1, d2 = data[:split_index], data[split_index:]

    d1.reset_index(drop=True, inplace=True)
    d2.reset_index(drop=True, inplace=True)

    return d1, d2


def to_sequences(seq_size, t1, t2):
    x = []
    y = []
    for i in range(len(t1) - seq_size):
        ta1 = t1[i:(i + seq_size)]
        ta2 = t2[i - 1 + seq_size]
        # ta1 = [[x] for x in ta1]
        x.append(ta1)
        y.append(ta2)

    # return x, y
    return np.array(x), np.array(y)


def balance_data(t1, t2):
    x = []
    y = []
    for i in range(len(t1)):
        if sum(t1[i]) > 0.2:
            x.append(t1[i])
            y.append(t2[i])

    return np.array(x), np.array(y)


def one_hot_encoding(target):
    result = []
    for i in target:
        cal = [0, 0, 0]
        cal[i] = 1
        result.append(cal)

    return np.array(result)
