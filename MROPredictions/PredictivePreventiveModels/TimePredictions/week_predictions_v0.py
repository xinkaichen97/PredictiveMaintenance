import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from collections import Counter


def read_data(filename, read_raw=False):
    if read_raw:
        data = pd.read_csv(filename, sep=',')
    else:
        data = pd.read_csv(filename, sep=',')
        print(data)
        # data = data[['CODE', 'YEAR', 'WEEK', 'TXN']]
        data = data[['CODE', 'WEEK', 'MONTH', 'QUARTER', 'TXN']]
        # data['YEAR'] -= 2014
        # data['YEAR'] += 356
        data['WEEK'] += 356
        data['MONTH'] += 409
        data['QUARTER'] += 421

    inputs = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    if read_raw:
        return inputs
    else:
        x = np.zeros((inputs.shape[0], 426))
        x[:, 0] = 1
        for i in range(inputs.shape[0]):
            for item in inputs.iloc[i, :]:
                x[i, int(item)] = 1
        return x, y


def get_error(inputs, raw_data, labels, model, output_file):
    pred = model.predict(inputs)
    # print(pred)
    pred = np.argmax(pred, axis=1)
    # print(pred)
    pred = [int(p) for p in pred]
    diffs = []
    zero_count, one_count = 0, 0
    correct_zero, correct_one = 0, 0
    for i in range(len(labels)):
        if labels[i] == 0:
            zero_count += 1
            if labels[i] == pred[i]:
                correct_zero += 1
        else:
            one_count += 1
            if labels[i] == pred[i]:
                correct_one += 1
    print('Zero accuracy:', (correct_zero / zero_count))
    print('One accuracy:', (correct_one / one_count))
    print('Total accuracy:', (correct_zero + correct_one) / len(labels))
    # with open(output_file, 'w', newline='') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(['ORG', 'ITEM', 'ACTIVITY_ITEM',
    #                      'ASSET_NUMBER', 'YEAR', 'WEEK', 'MONTH', 'QUARTER',
    #                      'TRUE', 'PREDICTED', 'DIFF'])
    #     for i in range(len(labels)):
    #         diff = 0 if labels[i] == pred[i] else 1
    #         diffs.append(diff)
    #         # diff_percent = abs(1 - (diff / labels[i]))
    #         # diffs.append(diff_percent)
    #         org = raw_data.iloc[i, 0]
    #         item = raw_data.iloc[i, 1]
    #         activity = raw_data.iloc[i, 2]
    #         asset = raw_data.iloc[i, 3]
    #         year = raw_data.iloc[i, 4]
    #         week = raw_data.iloc[i, 5]
    #         month = raw_data.iloc[i, 6]
    #         quarter = raw_data.iloc[i, 7]
    #         writer.writerow([org, item, activity, asset, year, week, month,
    #                          quarter, labels[i], pred[i], diff])
    # # for threshold in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    # #     get_accuracy(threshold, diffs)
    # print(1 - (sum(diffs) / len(diffs)))
    # print()


def get_accuracy(threshold, diffs):
    count = 0
    for i in range(len(diffs)):
        if threshold < diffs[i] < 1:
            count += 1
    print(threshold, (count / len(diffs)))


def get_item_counts(file):
    data = pd.read_csv(file).fillna(0)
    items = list(data['ITEM'])
    activities = list(data['ACTIVITY_ITEM'])
    assets = list(data['ASSET_NUMBER'])
    item_counts = dict(Counter(items))
    activity_counts = dict(Counter(activities))
    asset_counts = dict(Counter(assets))
    return item_counts, activity_counts, asset_counts


def get_dict(lst):
    dic = {item: i for i, item in enumerate(lst)}
    return dic


def get_mapped_df(df, org_dict, item_dict, activity_dict, asset_dict):
    # orgs
    orgs = list(df['ORGANIZATION_CODE'])
    mapped_orgs = [org_dict[org] for org in orgs]
    df['ORGANIZATION_CODE'] = mapped_orgs
    # items
    items = list(df['ITEM'])
    mapped_items = [item_dict[str(item)] for item in items]
    df['ITEM'] = mapped_items
    # activities
    activities = list(df['ACTIVITY_ITEM'])
    mapped_activities = [activity_dict[activity] for activity in activities]
    df['ACTIVITY_ITEM'] = mapped_activities
    # assets
    assets = list(df['ASSET_NUMBER'])
    mapped_assets = [asset_dict[asset] for asset in assets]
    df['ASSET_NUMBER'] = mapped_assets
    return df


def get_input_data(data, org_dict, item_dict, activity_dict, asset_dict):
    data = data[['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'WEEK', 'MONTH', 'QUARTER', 'TXN']]
    # data = data[['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'WEEK', 'TXN']]
    data['ORGANIZATION_CODE'] += 1
    data['ITEM'] += (1 + len(org_dict))
    data['ACTIVITY_ITEM'] += (1 + len(org_dict) + len(item_dict))
    data['ASSET_NUMBER'] += (1 + len(org_dict) + len(item_dict) + len(activity_dict))
    data['WEEK'] += (1 + len(org_dict) + len(item_dict) + len(activity_dict) + len(asset_dict))
    data['MONTH'] += (54 + len(org_dict) + len(item_dict) + len(activity_dict) + len(asset_dict))
    data['QUARTER'] += (66 + len(org_dict) + len(item_dict) + len(activity_dict) + len(asset_dict))
    inputs = data.iloc[:, :-1]
    total_features = 1 + len(org_dict) + len(item_dict) + len(activity_dict) + len(asset_dict) + 53 + 12 + 4
    print(total_features)
    x = np.zeros((inputs.shape[0], total_features + 1))
    x[:, 0] = 1  # bias term
    for i in range(inputs.shape[0]):
      for item in inputs.iloc[i, :]:
        # print(item)
        x[i, int(item)] = 1
    y = data.iloc[:, -1]
    return x, y


def get_input_with_codes(data, code_dict):
    # data = data.drop(columns=['YEAR'])
    data['CODE'] += 1
    data['WEEK'] += (1 + len(code_dict))
    data['MONTH'] += (54 + len(code_dict))
    data['QUARTER'] += (66 + len(code_dict))
    # data['TRANSACTION_QUANTITY'] *= -1
    inputs = data.iloc[:, :-1]
    total_features = 1 + len(code_dict) + 53 + 12 + 4
    print(total_features)
    x = np.zeros((inputs.shape[0], total_features + 1))
    x[:, 0] = 1  # bias term
    for i in range(inputs.shape[0]):
      for item in inputs.iloc[i, :]:
        x[i, int(item)] = 1
    y = data.iloc[:, -1]
    return x, y


def get_mapped_code(df, code_dict):
    orgs = list(df['ORGANIZATION_CODE'])
    items = list(df['ITEM'])
    activities = list(df['ACTIVITY_ITEM'])
    assets = list(df['ASSET_NUMBER'])
    mapped_codes = [code_dict[(str(o)+str(i)+str(a)+str(s))] for o, i, a, s in zip(orgs, items, activities, assets)]
    # mapped_orgs = [org_dict[org] for org in orgs]
    df['CODE'] = mapped_codes
    df = df[['CODE', 'WEEK', 'MONTH', 'QUARTER', 'TXN']]
    return df


def get_error_with_codes(inputs, raw_data, labels, model, output_file):
    pred = model.predict(inputs)
    # print(pred)
    pred = np.argmax(pred, axis=1)
    # print(pred)
    pred = [int(p) for p in pred]
    diffs = []
    zero_count, one_count = 0, 0
    correct_zero, correct_one = 0, 0
    for i in range(len(labels)):
        if labels[i] == 0:
            zero_count += 1
            if labels[i] == pred[i]:
                correct_zero += 1
        else:
            one_count += 1
            if labels[i] == pred[i]:
                correct_one += 1
    print('Zero accuracy:', (correct_zero / zero_count))
    print('One accuracy:', (correct_one / one_count))
    print('Total accuracy:', (correct_zero + correct_one) / len(labels))
    # with open(output_file, 'w', newline='') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(['ORG', 'ITEM', 'ACTIVITY_ITEM',
    #                      'ASSET_NUMBER', 'YEAR', 'WEEK', 'MONTH', 'QUARTER',
    #                      'TRUE', 'PREDICTED', 'DIFF'])
    #     zero_count, one_count = 0, 0
    #     correct_zero, correct_one = 0, 0
    #     for i in range(len(labels)):
    #
    #         diff = 0 if labels[i] == pred[i] else 1
    #         diffs.append(diff)
    #         # diff_percent = abs(1 - (diff / labels[i]))
    #         # diffs.append(diff_percent)
    #         org = raw_data.iloc[i, 0]
    #         item = raw_data.iloc[i, 1]
    #         activity = raw_data.iloc[i, 2]
    #         asset = raw_data.iloc[i, 3]
    #         year = raw_data.iloc[i, 4]
    #         week = raw_data.iloc[i, 5]
    #         month = raw_data.iloc[i, 6]
    #         quarter = raw_data.iloc[i, 7]
    #         writer.writerow([org, item, activity, asset, year, week, month,
    #                          quarter, labels[i], pred[i], diff])
    # # for threshold in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    # #     get_accuracy(threshold, diffs)
    # print(1 - (sum(diffs) / len(diffs)))
    # print()


if __name__ == '__main__':
    # total_data = pd.read_csv('TRX_1519_RANDOM_WEEKS_2.csv')
    # train_data = total_data[total_data['YEAR'] < 2019]
    # train_data = train_data[train_data['ORGANIZATION_CODE'] == 'FLO']

    train_data = pd.read_csv('TRX_1518_ALL_WEEKS.csv')
    # train_data = train_data[train_data['YEAR'].isin([2017, 2018])]
    train_data = train_data[train_data['ORGANIZATION_CODE'] == 'FLO']
    # print(train_data)
    test_data = pd.read_csv('TRX_19_ALL_WEEKS.csv')
    test_data = test_data[test_data['ORGANIZATION_CODE'] == 'FLO']

    # test_data = total_data[total_data['YEAR'] == 2019]
    # test_data = test_data[test_data['ORGANIZATION_CODE'] == 'FLO']
    total_data = pd.concat([train_data, test_data], ignore_index=True)
    total_data['ITEM'] = total_data['ITEM'].apply(str)
    total_data['ACTIVITY_ITEM'] = total_data['ACTIVITY_ITEM'].apply(str)
    total_data['ASSET_NUMBER'] = total_data['ASSET_NUMBER'].apply(str)
    items = sorted(set(list(total_data['ITEM'])))
    orgs = sorted(set(list(total_data['ORGANIZATION_CODE'])))
    activities = sorted(set(list(total_data['ACTIVITY_ITEM'])))
    assets = sorted(set(list(total_data['ASSET_NUMBER'])))
    org_dict = get_dict(orgs)
    item_dict = get_dict(items)
    activity_dict = get_dict(activities)
    asset_dict = get_dict(assets)
    print(org_dict)
    print(item_dict)
    print(activity_dict)
    print(asset_dict)


    # train_data = train_data[train_data['ORGANIZATION_CODE'] == 'FLO']
    # test_data = test_data[test_data['ORGANIZATION_CODE'] == 'FLO']
    # train_data = train_data[train_data['ORGANIZATION_CODE'].isin(['ENN', 'FLO', 'WIC'])]
    # test_data = test_data[test_data['ORGANIZATION_CODE'].isin(['ENN', 'FLO', 'WIC'])]
    # print(train_data)
    train_raw = train_data.copy()
    print(train_raw)
    test_raw = test_data.copy()

    # codes = [a + b + c + d for a, b, c, d in
    #          zip(list(total_data['ORGANIZATION_CODE']), list(total_data['ITEM']), list(total_data['ACTIVITY_ITEM']),
    #              list(total_data['ASSET_NUMBER']))]
    # codes = sorted(set(codes))
    # code_dict = get_dict(codes)
    # print(code_dict)
    # print(len(code_dict))
    # train_data = get_mapped_code(train_data, code_dict)
    # test_data = get_mapped_code(test_data, code_dict)
    # train_x, train_y = get_input_with_codes(train_data, code_dict)
    # test_x, test_y = get_input_with_codes(test_data, code_dict)

    train_data = get_mapped_df(train_data, org_dict, item_dict, activity_dict, asset_dict)
    print(train_data)
    test_data = get_mapped_df(test_data, org_dict, item_dict, activity_dict, asset_dict)
    print(test_data)

    train_x, train_y = get_input_data(train_data, org_dict, item_dict, activity_dict, asset_dict)
    test_x, test_y = get_input_data(test_data, org_dict, item_dict, activity_dict, asset_dict)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, input_dim=train_x.shape[1], activation='relu'),
        # tf.keras.layers.Dense(500, input_dim=train_x.shape[1], activation='relu'),
        tf.keras.layers.Dense(2),
        tf.keras.layers.Softmax()
    ])
    # model = tf.keras.models.load_model('model_classification_flo.h5')
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(train_x, train_y, epochs=1)
    # model.save('model_classification_flo.h5')
    # get_error_with_codes(train_x, train_raw, list(train_y), model, 'train_metric_classification_enn.csv')
    # get_error_with_codes(test_x, test_raw, list(test_y), model, 'test_metric_classification_enn.csv')
    get_error(train_x, train_raw, list(train_y), model, 'train_metric_classification_flo.csv')
    get_error(test_x, test_raw, list(test_y), model, 'test_metric_classification_flo.csv')
