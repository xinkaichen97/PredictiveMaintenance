import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter


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
    mapped_items = [item_dict[item] for item in items]
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


def get_mapped_code(df, code_dict):
    orgs = list(df['ORGANIZATION_CODE'])
    items = list(df['ITEM'])
    activities = list(df['ACTIVITY_ITEM'])
    assets = list(df['ASSET_NUMBER'])
    mapped_codes = [code_dict[(o + i + a + s)] for o, i, a, s in zip(orgs, items, activities, assets)]
    # mapped_orgs = [org_dict[org] for org in orgs]
    df['CODE'] = mapped_codes
    df = df[['CODE', 'YEAR', 'WEEK', 'MONTH', 'QUARTER', 'TRANSACTION_QUANTITY']]
    return df


def get_input_data(data, org_dict, item_dict, activity_dict, asset_dict):
    data = data[['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'WEEK', 'MONTH', 'QUARTER',
                 'TRANSACTION_QUANTITY']]
    data['ORGANIZATION_CODE'] += 1
    data['ITEM'] += (1 + len(org_dict))
    data['ACTIVITY_ITEM'] += (1 + len(org_dict) + len(item_dict))
    data['ASSET_NUMBER'] += (1 + len(org_dict) + len(item_dict) + len(activity_dict))
    data['WEEK'] += (1 + len(org_dict) + len(item_dict) + len(activity_dict) + + len(asset_dict))
    data['MONTH'] += (54 + len(org_dict) + len(item_dict) + len(activity_dict) + + len(asset_dict))
    data['QUARTER'] += (66 + len(org_dict) + len(item_dict) + len(activity_dict) + + len(asset_dict))
    data['TRANSACTION_QUANTITY'] *= -1
    inputs = data.iloc[:, :-1]
    total_features = 1 + len(org_dict) + len(item_dict) + len(activity_dict) + len(asset_dict) + 53 + 12 + 4
    print(total_features)
    x = np.zeros((inputs.shape[0], total_features + 1))
    x[:, 0] = 1  # bias term
    for i in range(inputs.shape[0]):
        for item in inputs.iloc[i, :]:
            x[i, int(item)] = 1
    y = data.iloc[:, -1]
    return x, y


def get_input_with_codes(data, code_dict):
    data = data.drop(columns=['YEAR'])
    data['CODE'] += 1
    data['WEEK'] += (1 + len(code_dict))
    data['MONTH'] += (54 + len(code_dict))
    data['QUARTER'] += (66 + len(code_dict))
    data['TRANSACTION_QUANTITY'] *= -1
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


def get_error(inputs, raw_data, labels, item_counts, activity_counts, asset_counts, model, output_file):
    pred = model.predict(inputs)
    pred = [float(p) for p in pred]
    diffs = []
    raw_data = raw_data[
        ['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'YEAR', 'WEEK', 'MONTH', 'QUARTER',
         'TRANSACTION_QUANTITY']]
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['ORGANIZATION_CODE', 'ITEM', 'ITEM_COUNT', 'ACTIVITY_ITEM', 'ACTIVITY_COUNT',
                         'ASSET_NUMBER', 'ASSET_COUNT', 'YEAR', 'WEEK', 'MONTH', 'QUARTER',
                         'TRANSACTION_QUANTITY', 'PREDICTED_QUANTITY', 'DIFF', 'DIFF_PERCENT'])
        for i in range(len(labels)):
            diff = abs(labels[i] - pred[i])
            diff_percent = abs(1 - (diff / labels[i]))
            diffs.append(diff_percent)
            org = raw_data.iloc[i, 0]
            item = raw_data.iloc[i, 1]
            item_count = item_counts[str(item)] if str(item) in item_counts else 0
            activity = raw_data.iloc[i, 2]
            activity_count = activity_counts[str(activity)] if str(activity) in activity_counts else 0
            asset = raw_data.iloc[i, 3]
            asset_count = asset_counts[str(asset)] if str(asset) in asset_counts else 0
            year = raw_data.iloc[i, 4]
            week = raw_data.iloc[i, 5]
            month = raw_data.iloc[i, 6]
            quarter = raw_data.iloc[i, 7]
            writer.writerow([org, item, item_count, activity, activity_count,
                             asset, asset_count, year, week, month, quarter,
                             labels[i], pred[i], diff, diff_percent])
    for threshold in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        get_accuracy(threshold, diffs)
    print()


def get_error_with_codes(inputs, raw_data, labels, model, output_file):
    pred = model.predict(inputs)
    pred = [float(p) for p in pred]
    diffs = []
    raw_data = raw_data[
        ['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'YEAR', 'WEEK', 'MONTH', 'QUARTER',
         'TRANSACTION_QUANTITY']]
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['ORGANIZATION_CODE', 'ITEM', 'ITEM_COUNT', 'ACTIVITY_ITEM', 'ACTIVITY_COUNT',
                         'ASSET_NUMBER', 'ASSET_COUNT', 'YEAR', 'WEEK', 'MONTH', 'QUARTER',
                         'TRANSACTION_QUANTITY', 'PREDICTED_QUANTITY', 'DIFF', 'DIFF_PERCENT'])


def get_accuracy(threshold, diffs):
    count = 0
    for i in range(len(diffs)):
        if threshold < diffs[i] < 1:
            count += 1
    print(threshold, (count / len(diffs)))


def get_item_counts(data):
    items = list(data['ITEM'])
    activities = list(data['ACTIVITY_ITEM'])
    assets = list(data['ASSET_NUMBER'])
    item_counts = dict(Counter(items))
    activity_counts = dict(Counter(activities))
    asset_counts = dict(Counter(assets))
    return item_counts, activity_counts, asset_counts


if __name__ == '__main__':
    train_data = pd.read_csv('TRX_ACTIVITY_15_18.csv').fillna(0)
    test_data = pd.read_csv('TRX_ACTIVITY_19.csv').fillna(0)
    train_raw = train_data.copy()
    test_raw = test_data.copy()
    print(train_data)
    total_data = pd.concat([train_data, test_data], ignore_index=True)
    items = sorted(set(list(total_data['ITEM'])))
    orgs = sorted(set(list(total_data['ORGANIZATION_CODE'])))
    activities = sorted(set(list(total_data['ACTIVITY_ITEM'])))
    assets = sorted(set(list(total_data['ASSET_NUMBER'])))
    org_dict = get_dict(orgs)
    item_dict = get_dict(items)
    activity_dict = get_dict(activities)
    asset_dict = get_dict(assets)

    # codes = [a+b+c+d for a,b,c,d in zip(list(total_data['ORGANIZATION_CODE']), list(total_data['ITEM']), list(total_data['ACTIVITY_ITEM']),
    #            list(total_data['ASSET_NUMBER']))]
    # codes = sorted(set(codes))
    # code_dict = get_dict(codes)
    # print(code_dict)
    # print(len(code_dict))
    # train_data = get_mapped_code(train_data, code_dict)
    # test_data = get_mapped_code(test_data, code_dict)
    # print(train_data)
    # train_data = train_data.groupby(['CODE', 'YEAR', 'WEEK', 'MONTH', 'QUARTER'], as_index=False).sum()
    # print(train_data)
    # train_x, train_y = get_input_with_codes(train_data, code_dict)
    # print(train_x)
    # print(train_x.shape)
    # assert 1 == 0
    # test_x, test_y = get_input_with_codes(test_data, code_dict)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(8000, input_dim=train_x.shape[1], activation='relu'),
    #     tf.keras.layers.Dense(8000, input_dim=train_x.shape[1], activation='relu'),
    #     tf.keras.layers.Dense(1)
    # ])
    # model = tf.keras.models.load_model('model_1518_code_e10.h5')
    # model.compile(optimizer='adam', loss='mae')
    # model.fit(train_x, train_y, epochs=10)
    # model.save('model_1518_code_e10.h5')

    # item_counts, activity_counts, asset_counts = get_item_counts(total_data)
    # get_error(train_x, train_raw, train_y, item_counts, activity_counts, asset_counts, model, 'train_metric_1518.csv')
    # get_error(test_x, test_raw, test_y, item_counts, activity_counts, asset_counts, model, 'test_metric_1518.csv')

    train_data = get_mapped_df(train_data, org_dict, item_dict, activity_dict, asset_dict)
    # print(train_data)
    test_data = get_mapped_df(test_data, org_dict, item_dict, activity_dict, asset_dict)

    train_data = train_data.groupby(['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER',
                                     'YEAR', 'WEEK', 'MONTH', 'QUARTER'], as_index=False).sum()
    print(train_data)
    train_x, train_y = get_input_data(train_data, org_dict, item_dict, activity_dict, asset_dict)
    test_x, test_y = get_input_data(test_data, org_dict, item_dict, activity_dict, asset_dict)
    print(train_x)
    print(train_x.shape)
    print(train_y)
    # assert 1 == 0
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(5000, input_dim=train_x.shape[1], activation='relu'),
    #     tf.keras.layers.Dense(5000, input_dim=train_x.shape[1], activation='relu'),
    #     tf.keras.layers.Dense(1)
    # ])
    model = tf.keras.models.load_model('model_1518_e15.h5')
    # model.compile(optimizer='adam', loss='mae')
    # model.fit(train_x, train_y, epochs=5)
    # model.save('model_1518_e20.h5')

    item_counts, activity_counts, asset_counts = get_item_counts(total_data)
    get_error(train_x, train_raw, train_y, item_counts, activity_counts, asset_counts, model, 'train_metric_1518.csv')
    get_error(test_x, test_raw, test_y, item_counts, activity_counts, asset_counts, model, 'test_metric_1518.csv')
