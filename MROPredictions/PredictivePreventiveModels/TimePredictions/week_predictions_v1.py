import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import re
from collections import Counter
from train_classification import get_dict, get_mapped_df


def get_error(pred, raw_data, labels, model, output_file):
    diffs = []
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['ORG', 'ITEM', 'ACTIVITY_ITEM',
                         'ASSET_NUMBER', 'YEAR', 'WEEK', 'MONTH', 'QUARTER',
                         'TRUE', 'PREDICTED', 'DIFF'])
        for i in range(len(labels)):
            diff = 0 if labels[i] == pred[i] else 1
            diffs.append(diff)
            org = raw_data.iloc[i, 0]
            item = raw_data.iloc[i, 1]
            activity = raw_data.iloc[i, 2]
            asset = raw_data.iloc[i, 3]
            year = raw_data.iloc[i, 4]
            week = raw_data.iloc[i, 5]
            month = raw_data.iloc[i, 6]
            quarter = raw_data.iloc[i, 7]
            writer.writerow([org, item, activity, asset, year, week, month,
                             quarter, labels[i], pred[i], diff])


def get_input_data(data, org_dict, item_dict, activity_dict, asset_dict):
    data = data[['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'WEEK']]
    data['ORGANIZATION_CODE'] += 1
    data['ITEM'] += (1 + len(org_dict))
    data['ACTIVITY_ITEM'] += (1 + len(org_dict) + len(item_dict))
    data['ASSET_NUMBER'] += (1 + len(org_dict) + len(item_dict) + len(activity_dict))
    # data['WEEK'] += (1 + len(org_dict) + len(item_dict) + len(activity_dict) + len(asset_dict))
    inputs = data.iloc[:, :-1]
    total_features = 1 + len(org_dict) + len(item_dict) + len(activity_dict) + len(asset_dict)
    print(total_features)
    x = np.zeros((inputs.shape[0], total_features + 1))
    x[:, 0] = 1  # bias term
    for i in range(inputs.shape[0]):
        for item in inputs.iloc[i, :]:
            x[i, int(item)] = 1
    weeks = list(data.iloc[:, -1])
    y = np.zeros((inputs.shape[0], 53))
    for i in range(len(weeks)):
        week = weeks[i]
        y[i, week - 1] = 1
    print(y)
    print(y.shape)
    return x, y


def get_predictions(inputs, model, weeks, labels=None):
    pred = model.predict(inputs)
    pred_binary = []
    for i in range(len(weeks)):
        week = weeks[i]
        outputs = np.sort(pred[i, :])
        thres = outputs[-25]
        first_week = 0
        # print(outputs, thres)
        output = pred[i, week - 1]
        if output > thres:
            pred_binary.append(1)
        else:
            pred_binary.append(0)
    print(pred_binary)

    if labels is not None:
        zero_count, one_count = 0, 0
        correct_zero, correct_one = 0, 0
        for i in range(len(labels)):
            if labels[i] == 0:
                zero_count += 1
                if labels[i] == pred_binary[i]:
                    correct_zero += 1
            else:
                one_count += 1
                if labels[i] == pred_binary[i]:
                    correct_one += 1
        print('Zero accuracy:', (correct_zero / (zero_count + 1e-5)))
        print('One accuracy:', (correct_one / (one_count + 1e-5)))
        print('Total accuracy:', (correct_zero + correct_one) / len(labels))

    print(sum(pred_binary), len(pred_binary), sum(pred_binary) / len(pred_binary))
    return pred_binary


def macro_f1(y, y_hat, thresh=0.0145):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost


if __name__ == '__main__':
    org = 'FLO'
    train_data = pd.read_csv('TRX_ACTIVITY_15_19_FLO.csv')
    train_data = train_data[train_data['YEAR'] < 2019]
    train_data = train_data[train_data['ACTIVITY_COUNT'] >= 20]
    test_data = pd.read_csv('TRX_ACTIVITY_15_19_FLO.csv')
    test_data = test_data[test_data['YEAR'] == 2019]
    # activity = list(total_data['ACTIVITY_ITEM'])
    # cycles = []
    # count = 0
    # for a in activity:
    #     words = a.split('-')
    #     if len(words) <= 2:
    #         words = a.split(' ')
    #     found = False
    #     for word in reversed(words):
    #         if re.search('^[0-9]+\s?[A-Z]+', word) \
    #                 or word in ['SEMIANNUAL', 'SEMIANNUALLY', 'MONTH', 'BI ANNUAL', 'ANNUALLY', 'MONTHLY', 'WEEKLY', 'OPMONTHLY', 'YEARLY']:
    #         # if any(ch.isdigit() for ch in word) and (('HR' in word) or ('D' in word) or ):
    #             cycles.append(word)
    #             found = True
    #             break
    #     if not found:
    #         cycles.append('N/A')
    #         count += 1
    # print(cycles)
    # print(count, len(cycles))
    # total_data['CYCLE'] = cycles
    # total_data.to_csv('TRX_ACTIVITY_15_19_2.csv')
    # assert 1 == 0

    # train_data = total_data[total_data['YEAR'] < 2019]

    test_data_with_zeros = pd.read_csv('TRX_19_ALL_WEEKS.csv')
    test_data_with_zeros = test_data_with_zeros[test_data_with_zeros['ORGANIZATION_CODE'] == org]
    test_data_with_zeros = test_data_with_zeros.drop_duplicates()

    # test_data_with_zeros['ITEM'] = test_data_with_zeros['ITEM'].apply(str)
    # train_items = list(train_data['ITEM'])
    # train_activities = list(train_data['ACTIVITY_ITEM'])
    # train_assets = list(train_data['ASSET_NUMBER'])
    # print(train_assets)
    # test_data_with_zeros = test_data_with_zeros[test_data_with_zeros['ITEM'].isin(train_items)]
    # test_data_with_zeros = test_data_with_zeros[test_data_with_zeros['ACTIVITY_ITEM'].isin(train_activities)]
    # test_data_with_zeros = test_data_with_zeros[test_data_with_zeros['ASSET_NUMBER'].isin(train_assets)]
    train_com = list(train_data['ITEM_ACTIVITY_ASSET'])
    test_data_with_zeros = test_data_with_zeros[test_data_with_zeros['ITEM_ACTIVITY_ASSET'].isin(train_com)]

    train_data = train_data[train_data['ORGANIZATION_CODE'] == org]
    test_data = test_data[test_data['ORGANIZATION_CODE'] == org]
    train_counts = list(train_data['ACTIVITY_COUNT'])
    test_counts_wz = list(test_data_with_zeros['ACTIVITY_COUNT'])
    train_cycles = list(train_data['CYCLE'])
    test_cycles_wz = list(test_data_with_zeros['CYCLE'])
    print(test_counts_wz)
    print(test_cycles_wz)
    train_weeks = list(train_data['WEEK'])
    test_weeks = list(test_data['WEEK'])
    test_weeks_with_zeros = list(test_data_with_zeros['WEEK'])
    labels_with_zeros = list(test_data_with_zeros['TXN'])

    train_raw = train_data.copy()
    test_raw_wz = test_data_with_zeros.copy()

    total_data = pd.concat([train_data, test_data], ignore_index=True)
    total_data['ITEM'] = total_data['ITEM'].apply(str)
    total_data['ACTIVITY_ITEM'] = total_data['ACTIVITY_ITEM'].apply(str)
    total_data['ASSET_NUMBER'] = total_data['ASSET_NUMBER'].apply(str)
    items = sorted(set(list(total_data['ITEM'])))
    # print(items)
    orgs = sorted(set(list(total_data['ORGANIZATION_CODE'])))
    activities = sorted(set(list(total_data['ACTIVITY_ITEM'])))
    assets = sorted(set(list(total_data['ASSET_NUMBER'])))
    org_dict = get_dict(orgs)
    item_dict = get_dict(items)
    activity_dict = get_dict(activities)
    asset_dict = get_dict(assets)
    # test_data_with_zeros = test_data_with_zeros[test_data_with_zeros['ITEM'].isin(items)]

    train_data = get_mapped_df(train_data, org_dict, item_dict, activity_dict, asset_dict)
    # print(train_data)
    test_data = get_mapped_df(test_data, org_dict, item_dict, activity_dict, asset_dict)
    # print(test_data)
    test_data_with_zeros = get_mapped_df(test_data_with_zeros, org_dict, item_dict, activity_dict, asset_dict)
    # print(test_data_with_zeros)

    train_x, train_y = get_input_data(train_data, org_dict, item_dict, activity_dict, asset_dict)
    test_x, test_y = get_input_data(test_data, org_dict, item_dict, activity_dict, asset_dict)
    test_x_wz, test_y_wz = get_input_data(test_data_with_zeros, org_dict, item_dict, activity_dict, asset_dict)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(300, input_dim=train_x.shape[1], activation='relu'),
        # tf.keras.layers.Dense(3000, input_dim=train_x.shape[1], activation='relu'),
        tf.keras.layers.Dense(53, activation='sigmoid')
    ])
    # model = tf.keras.models.load_model('model_classification_nonzero_flo.h5')
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
                  loss=tf.keras.losses.binary_crossentropy)
    # loss=macro_soft_f1,
    # metrics=[macro_f1])
    model.fit(train_x, train_y, epochs=1)
    # model.save('model_classification_nonzero_flo.h5')
    preds_train = model.predict(train_x)
    # preds_test = model.predict(test_x)
    # print(preds_train[0, :])
    # print(np.amax(preds_train, axis=1))
    # print(preds_test)
    # print(np.amax(preds_test, axis=1))
    pred_train = get_predictions(train_x, model, train_weeks, train_counts, train_cycles)
    # get_predictions(test_x, model, test_weeks)
    pred_test_wz = get_predictions(test_x_wz, model, test_weeks_with_zeros, test_counts_wz,
                                   test_cycles_wz, labels_with_zeros)

    # get_error(train_x, train_raw, labels_with_zeros, model, 'train_metric_classification_flo.csv')
    get_error(pred_test_wz, test_raw_wz, labels_with_zeros, model, 'test_metric_classification_flo.csv')
