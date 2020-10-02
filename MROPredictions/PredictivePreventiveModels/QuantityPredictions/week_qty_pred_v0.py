import pandas as pd
import numpy as np
import tensorflow as tf
from train_PPM import get_dict, get_mapped_df


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


def get_input_data(data, org_dict, item_dict, asset_dict):
    data = data[['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'WEEK',
                 'TRANSACTION_QUANTITY']]
    data['ORGANIZATION_CODE'] += 1
    data['ITEM'] += (1 + len(org_dict))
    data['ACTIVITY_ITEM'] += (1 + len(org_dict) + len(item_dict))
    data['ASSET_NUMBER'] += (1 + len(org_dict) + len(item_dict) + len(activity_dict))
    data['WEEK'] += (1 + len(org_dict) + len(item_dict) + len(activity_dict) + + len(asset_dict))
    data['TRANSACTION_QUANTITY'] *= -1
    inputs = data.iloc[:, :-1]
    total_features = 1 + len(org_dict) + len(item_dict) + len(activity_dict) + len(asset_dict) + 53
    print(total_features)
    x = np.zeros((inputs.shape[0], total_features + 1))
    x[:, 0] = 1  # bias term
    for i in range(inputs.shape[0]):
        for item in inputs.iloc[i, :]:
            x[i, int(item)] = 1
    y = data.iloc[:, -1]
    return x, y


if __name__ == '__main__':
    train_data = pd.read_csv('TRX_ACTIVITY_15_18.csv').fillna(0)
    test_data = pd.read_csv('TRX_ACTIVITY_19.csv').fillna(0)
    selected_orgs = ['BED', 'EFF', 'FLO', 'GAR', 'WAC']
    train_data = train_data[train_data['ORGANIZATION_CODE'].isin(selected_orgs)]
    test_data = test_data[test_data['ORGANIZATION_CODE'].isin(selected_orgs)]
    train_raw = train_data.copy()
    test_raw = test_data.copy()
    total_data = pd.concat([train_data, test_data], ignore_index=True)

    items = sorted(set(list(total_data['ITEM'])))
    orgs = sorted(set(list(total_data['ORGANIZATION_CODE'])))
    activities = sorted(set(list(total_data['ACTIVITY_ITEM'])))
    assets = sorted(set(list(total_data['ASSET_NUMBER'])))
    org_dict = get_dict(orgs)
    item_dict = get_dict(items)
    activity_dict = get_dict(activities)
    asset_dict = get_dict(assets)

    train_data = get_mapped_df(train_data, org_dict, item_dict, activity_dict, asset_dict)
    # print(train_data)
    test_data = get_mapped_df(test_data, org_dict, item_dict, activity_dict, asset_dict)

    # train_data = train_data.groupby(['ORGANIZATION_CODE', 'ITEM', 'ASSET_NUMBER',
    #                                  'WEEK'], as_index=False).sum()

    train_x, train_y = get_input_data(train_data, org_dict, item_dict, asset_dict)
    test_x, test_y = get_input_data(test_data, org_dict, item_dict, asset_dict)

    print(train_x, train_y)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3000, input_dim=train_x.shape[1], activation='relu'),
        tf.keras.layers.Dense(500, input_dim=train_x.shape[1], activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    # model = tf.keras.models.load_model('model_week_qty_e10.h5')
    model.compile(optimizer='adam', loss='mae')
    model.fit(train_x, train_y, epochs=10)
    # model.save('model_week_qty_e20.h5')

    pred_test = model.predict(test_x)
    test_raw['TRANSACTION_QUANTITY'] = test_raw['TRANSACTION_QUANTITY'] * -1
    test_raw['PRED_QTY'] = pred_test
    test_raw['DIFF'] = abs(test_raw['TRANSACTION_QUANTITY'] - test_raw['PRED_QTY'])
    test_raw['DIFF_PERCENT'] = test_raw['DIFF'] / test_raw['TRANSACTION_QUANTITY']
    print(test_raw[test_raw['DIFF_PERCENT'] < 0.1].shape[0])
    print(test_raw[test_raw['DIFF_PERCENT'] < 0.2].shape[0])
    print(test_raw[test_raw['DIFF_PERCENT'] < 0.3].shape[0])
    print(test_raw)
    # test_raw.to_csv('TRX_WEEK_QTY_TEST_e20.csv', index=False)
