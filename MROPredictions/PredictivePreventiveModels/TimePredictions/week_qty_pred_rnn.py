import numpy as np
import pandas as pd
import csv
import datetime
from cond_rnn import ConditionalRNN
# from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM
import tensorflow as tf
# from tensorflow.contrib.rnn.python.ops.rnn_cell import PhasedLSTMCell


def get_training_data(dataset, start_index, end_index, history_size):
    data = []
    labels = []
    if end_index is None:
        end_index = len(dataset)
    # inputs = dataset[['DAY', 'TRANSACTION_QUANTITY']]
    inputs = dataset[['TRANSACTION_QUANTITY']]
    # inputs['DAY'] = tf.cast(inputs['DAY'], tf.float32)
    for i in range(start_index + history_size, end_index):
        # padded_inputs = np.zeros((inputs.shape[0], 2))
        padded_inputs = np.zeros((inputs.shape[0], 1))
        padded_inputs[0:i] = np.array(inputs.iloc[0:i, :])
        # data.append(np.reshape(np.array(padded_inputs), (inputs.shape[0], 2)))
        data.append(np.reshape(np.array(padded_inputs), (inputs.shape[0], 1)))
        # labels.append(inputs.iloc[i, 1])
        labels.append(inputs.iloc[i, 0])
    data = np.array(data)
    print(data.shape)
    labels = np.array(labels)
    return data, labels


def get_error(data, pred, output_file):
    pred_dict = {int(day): qty for day, qty in pred}
    # data = data[['DAY', 'TRANSACTION_QUANTITY']]
    days = list(data['DAY'])
    qtys = list(data['TRANSACTION_QUANTITY'])
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['day', 'y_actual', 'y_pred', 'diff', 'diff%'])
        for i in range(10, data.shape[0]):
            day = int(days[i])
            qty_actual = float(qtys[i])
            if day in pred_dict.keys():
                qty_pred = float(pred_dict[day])
            else:
                if (day - 1) in pred_dict.keys():
                    if (day + 1) in pred_dict.keys():
                        qty_pred = (float(pred_dict[day - 1]) + float(pred_dict[day + 1])) / 2
                    else:
                        qty_pred = float(pred_dict[day - 1])
                else:
                    qty_pred = float(pred_dict[day + 1])
            diff = abs(qty_actual - qty_pred)
            diff_pct = abs(1 - (diff / qty_actual))
            writer.writerow([day, qty_actual, qty_pred, diff, diff_pct])


def get_predictions(inputs, end_date, model):
    inputs = inputs[['DAY', 'TRANSACTION_QUANTITY']]
    timesteps = inputs.shape[0]
    last_date = inputs.iloc[-1, 0]
    print(last_date)
    inputs = inputs[['TRANSACTION_QUANTITY']]
    inputs = [inputs.values]
    # inputs = np.reshape(inputs, (1, timesteps, 2))
    inputs = np.reshape(inputs, (1, timesteps, 1))
    print(inputs)
    predictions = []
    while last_date < end_date:
        pred = model.predict(inputs)
        pred = float(pred)
        # print(pred)
        new_date = last_date + 2
        # x = np.append(inputs[0], [[new_date, pred]], axis=0)
        x = np.append(inputs[0], [[pred]], axis=0)
        inputs = np.array([x])
        # print(inputs)
        predictions.append([new_date, pred])
        last_date = new_date
    return predictions


def get_train_predictions(inputs, model):
    inputs = inputs[['DAY', 'TRANSACTION_QUANTITY']]
    predictions = []
    for i in range(10, inputs.shape[0]):
        current_inputs = inputs.iloc[:i+1, :]
        current_date = current_inputs.iloc[-1, 0]
        current_inputs = inputs.iloc[:i, -1]
        current_inputs = [current_inputs.values]
        # current_inputs = np.reshape(current_inputs, (1, i, 2))
        current_inputs = np.reshape(current_inputs, (1, i, 1))
        pred = model.predict(current_inputs)
        pred = float(pred)
        # print(inputs)
        predictions.append([current_date, pred])
    return predictions


if __name__ == '__main__':
    # read data
    train_data = pd.read_csv('ISSUE_TRAIN_EFF_152194.csv')
    test_data = pd.read_csv('ISSUE_TEST_EFF_152194.csv')
    # no need to include org and item
    train_x = train_data.iloc[:, 2:]
    # convert to datetime
    train_x['TRANSACTION_DATE'] = pd.to_datetime(train_x['TRANSACTION_DATE'].astype(str))
    train_end_date = train_x['DAY'].max()

    # process test data
    test_x = test_data.iloc[:, 2:]
    test_x['TRANSACTION_DATE'] = pd.to_datetime(test_x['TRANSACTION_DATE'].astype(str))
    test_labels = test_x['TRANSACTION_QUANTITY']
    test_end_date = test_x['DAY'].max()

    # get inputs
    past_history = 10
    train_inputs, train_labels = get_training_data(train_x, 0, None, past_history)
    print(train_inputs)
    print(train_labels)
    # model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Masking(mask_value=0.0, input_shape=train_inputs.shape[-2:]),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    model.fit(train_inputs, train_labels, epochs=500)
    model.save('model.h5')
    # train_pred = get_predictions(train_x.iloc[:past_history, :], train_end_date, model)
    train_pred = get_train_predictions(train_x, model)
    print(train_pred)
    get_error(train_x, train_pred, 'train_metric.csv')
    test_pred = get_predictions(train_x, test_end_date, model)
    print(test_pred)
    get_error(test_x, test_pred, 'test_metric.csv')
