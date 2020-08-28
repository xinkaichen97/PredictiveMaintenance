import pandas as pd
import numpy as np
import tensorflow as tf


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
    data = data[['ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'COUNT']]
    data['ITEM'] += len(org_dict)
    data['ACTIVITY_ITEM'] += (len(org_dict) + len(item_dict))
    data['ASSET_NUMBER'] += (len(org_dict) + len(item_dict) + len(activity_dict))
    inputs = data.iloc[:, :-1]
    total_features = 1 + len(item_dict) + len(activity_dict) + len(asset_dict)
    print(total_features)
    x = np.zeros((inputs.shape[0], total_features + 1))
    x[:, 0] = 1  # bias term
    for i in range(inputs.shape[0]):
        for item in inputs.iloc[i, :]:
            x[i, int(item)] = 1
    y = data.iloc[:, -1]
    print(x, y)
    return x, y


def get_mapped_df_combo(data, iaa_dict):
    iaas = data['ITEM_ACTIVITY_ASSET']
    iaa_mapped = [iaa_dict[iaa] for iaa in iaas]
    data['ITEM_ACTIVITY_ASSET'] = iaa_mapped
    return data


def get_input_data_combo(data, iaa_dict):
    data = data[['ITEM_ACTIVITY_ASSET', 'COUNT']]
    inputs = data.iloc[:, :-1]
    total_features = 1 + len(iaa_dict)
    print(total_features)
    x = np.zeros((inputs.shape[0], total_features + 1))
    x[:, 0] = 1  # bias term
    for i in range(inputs.shape[0]):
        for item in inputs.iloc[i, :]:
            x[i, int(item)] = 1
    y = data.iloc[:, -1]
    print(x, y)
    return x, y


if __name__ == '__main__':
    org = 'FLO'
    total_data = pd.read_csv('TRX_ACTIVITY_15_19_FLO.csv')
    total_data = total_data[['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'YEAR']]
    total_data['COUNT'] = 1
    total_data['ITEM'] = total_data['ITEM'].apply(str)
    total_data['ACTIVITY_ITEM'] = total_data['ACTIVITY_ITEM'].apply(str)
    total_data['ASSET_NUMBER'] = total_data['ASSET_NUMBER'].apply(str)

    total_data = total_data.groupby(['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER',
                                     'YEAR'], as_index=False).sum()
    total_data['ITEM_ACTIVITY_ASSET'] = total_data['ITEM'] + total_data['ACTIVITY_ITEM'] + total_data['ASSET_NUMBER']
    print(total_data)
    year_count = total_data.groupby(['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER'],
                                    as_index=False).count()
    year_count = year_count[year_count['YEAR'] >= 4]
    year_count['ITEM_ACTIVITY_ASSET'] = year_count['ITEM'] + year_count['ACTIVITY_ITEM'] + year_count['ASSET_NUMBER']
    total_data = total_data[total_data['ITEM_ACTIVITY_ASSET'].isin(year_count['ITEM_ACTIVITY_ASSET'])]
    print(total_data)
    train_data = total_data[total_data['YEAR'] < 2019]
    test_data = total_data[total_data['YEAR'] == 2019]
    train_data = train_data[train_data['ITEM_ACTIVITY_ASSET'].isin(test_data['ITEM_ACTIVITY_ASSET'])]
    total_data = pd.concat([train_data, test_data], ignore_index=True)
    print(test_data)

    items = sorted(set(list(total_data['ITEM'])))
    orgs = sorted(set(list(total_data['ORGANIZATION_CODE'])))
    activities = sorted(set(list(total_data['ACTIVITY_ITEM'])))
    assets = sorted(set(list(total_data['ASSET_NUMBER'])))
    iaas = sorted(set(list(total_data['ITEM_ACTIVITY_ASSET'])))
    org_dict = get_dict(orgs)
    item_dict = get_dict(items)
    activity_dict = get_dict(activities)
    asset_dict = get_dict(assets)
    iaa_dict = get_dict(iaas)

    train_data = get_mapped_df(train_data, org_dict, item_dict, activity_dict, asset_dict)
    test_data = get_mapped_df(test_data, org_dict, item_dict, activity_dict, asset_dict)
    train_x, train_y = get_input_data(train_data, org_dict, item_dict, activity_dict, asset_dict)
    test_x, test_y = get_input_data(test_data, org_dict, item_dict, activity_dict, asset_dict)

    # train_data = get_mapped_df_combo(train_data, iaa_dict)
    # test_data = get_mapped_df_combo(test_data, iaa_dict)
    # train_x, train_y = get_input_data_combo(train_data, iaa_dict)
    # test_x, test_y = get_input_data_combo(test_data, iaa_dict)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, input_dim=train_x.shape[1], activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    model.fit(train_x, train_y, epochs=50)

    pred_test = model.predict(test_x)
    test_data['PRED_COUNT'] = pred_test
    test_data['DIFF'] = abs(test_data['COUNT'] - test_data['PRED_COUNT'])
    test_data['DIFF_PERCENT'] = test_data['DIFF'] / test_data['COUNT']
    print(test_data[test_data['DIFF_PERCENT'] < 0.1].shape[0])
    print(test_data)
    # test_data.to_csv('TRX_YEAR_COUNT_FLO_TEST_NN.csv', index=False)
    # total_data.to_csv('TRX_YEAR_COUNT_1519_FLO_2.csv', index=False)
    # print(total_data[total_data['COUNT'] > 1])
