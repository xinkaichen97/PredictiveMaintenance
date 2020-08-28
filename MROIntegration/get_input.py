import numpy as np


# indices start at 1 (0 reserved for 'none of any class')
def get_dict(lst):
    dic = {item: i for i, item in enumerate(lst, start=1)}
    return dic


# generate dictionaries from data
def get_dicts(total_data, program):
    # CRM does not have activity
    if program in ['CRM1', 'CRM2']:
        activity_dict = None
    else:
        activities = sorted(set(list(total_data.loc[total_data.ACTIVITY_ITEM.notnull(), 'ACTIVITY_ITEM'])))
        activity_dict = get_dict(activities)
    items = sorted(set(list(total_data['ITEM'])))
    orgs = sorted(set(list(total_data['ORGANIZATION_CODE'])))
    assets = sorted(set(list(total_data['ASSET_NUMBER'])))
    org_dict = get_dict(orgs)
    item_dict = get_dict(items)
    asset_dict = get_dict(assets)
    return org_dict, item_dict, activity_dict, asset_dict


# read dictionary from file
def read_dict(filename):
    try:
        with open(filename, 'r') as f:
            items = [line.rstrip('\n') for line in f]
            items_dict = {item.split(' ', 1)[1]: int(item.split(' ', 1)[0]) for item in items}
            return items_dict
    except OSError:
        return None


# read all dictionaries
def read_dicts(org_dict_name, item_dict_name, activity_dict_name, asset_dict_name):
    org_dict = read_dict(org_dict_name)
    item_dict = read_dict(item_dict_name)
    activity_dict = read_dict(activity_dict_name)
    asset_dict = read_dict(asset_dict_name)
    return org_dict, item_dict, activity_dict, asset_dict


# input: data after mapping the orgs and other attributes
# output: input data for the ML model
def get_input_data(data, org_dict, item_dict, activity_dict, asset_dict, program, total_features=None):
    data = data.fillna(0)
    # filter different columns for different programs
    if program == 'PPM1':
        data = data[['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'WEEK_NUMBER',
                     'TRANSACTION_QUANTITY']]
    elif program == 'PPM2':
        data = data[['ORGANIZATION_CODE', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER', 'YEAR_NUMBER',
                     'TRX_COUNT']]
    elif program == 'CRM1':
        data = data[['ORGANIZATION_CODE', 'ITEM', 'ASSET_NUMBER', 'WEEK_NUMBER',
                     'TRANSACTION_QUANTITY']]
    else:
        data = data[['ORGANIZATION_CODE', 'ITEM', 'ASSET_NUMBER', 'YEAR_NUMBER',
                     'TRX_COUNT']]
    # create an one-hot encoding for each training example
    # add bias term and use 0 for 'none of any classes' (indices from dict starts with 1)
    org_dict_len = 0
    if org_dict is not None:
        data['ORGANIZATION_CODE'] += 1
        org_dict_len = 1 + len(org_dict)
    item_dict_len = 0
    if item_dict is not None:
        data['ITEM'] += (1 + org_dict_len)
        item_dict_len = 1 + len(item_dict)
    activity_dict_len = 0
    if activity_dict is not None:
        data['ACTIVITY_ITEM'] += (1 + org_dict_len + item_dict_len)
        activity_dict_len = 1 + len(activity_dict)
    asset_dict_len = 0
    if asset_dict is not None:
        data['ASSET_NUMBER'] += (1 + org_dict_len + item_dict_len + activity_dict_len)
        asset_dict_len = 1 + len(asset_dict)
    # changes based on different programs
    if program in ['PPM1', 'CRM1']:
        data['TRANSACTION_QUANTITY'] *= -1
        data['WEEK_NUMBER'] += (1 + org_dict_len + item_dict_len + activity_dict_len + asset_dict_len)
    inputs = data.iloc[:, :-1]
    # if the number of features is not given (meaning creating a new model), it will be computed
    if total_features is None:
        if program in ['PPM1', 'CRM1']:
            total_features = 1 + org_dict_len + item_dict_len + activity_dict_len + asset_dict_len + 54
        else:
            total_features = 1 + org_dict_len + item_dict_len + activity_dict_len + asset_dict_len
    x = np.zeros((inputs.shape[0], total_features))
    x[:, 0] = 1  # add bias term
    # create one-hot encoding
    try:
        for i in range(inputs.shape[0]):
            for item in inputs.iloc[i, :]:
                x[i, int(item)] = 1
    except ValueError:
        x = None
    y = data.iloc[:, -1]
    return x, y, total_features


# get data mapping using dictionaries in order to pass into the ML model
# for not recognized values, will give 0
def get_mapped_df(df, org_dict, item_dict, activity_dict, asset_dict):
    # copy the data so that the original one remains unchanged
    df_copy = df.copy()
    # orgs
    if org_dict is not None:
        orgs = list(df_copy['ORGANIZATION_CODE'])
        mapped_orgs = [org_dict.get(org, 0) if org is not None else 0 for org in orgs]
        df_copy['ORGANIZATION_CODE'] = mapped_orgs
    else:
        df_copy['ORGANIZATION_CODE'] = 0
    # items
    if item_dict is not None:
        items = list(df_copy['ITEM'])
        mapped_items = [item_dict.get(item, 0) if item is not None else 0 for item in items]
        df_copy['ITEM'] = mapped_items
    else:
        df_copy['ITEM'] = 0
    # activities; can accept None
    if activity_dict is not None:
        activities = list(df_copy['ACTIVITY_ITEM'])
        mapped_activities = [activity_dict.get(activity, 0) if activity is not None else 0 for activity in activities]
        df_copy['ACTIVITY_ITEM'] = mapped_activities
    # assets
    if asset_dict is not None:
        assets = list(df_copy['ASSET_NUMBER'])
        mapped_assets = [asset_dict.get(asset, 0) if asset is not None else 0 for asset in assets]
        df_copy['ASSET_NUMBER'] = mapped_assets
    else:
        df_copy['ASSET_NUMBER'] = 0
    return df_copy


# save dictionary as file
def save_dict(dic, file_name):
    with open(file_name, 'w') as f:
        for item, index in dic.items():
            f.write(str(index) + ' ' + str(item) + '\n')


# for PPM2 and CRM2, group the data by transaction count
def group_data_by_count(data, program):
    if program == 'PPM2':
        data['TRX_COUNT'] = 1
        data = data[['ORGANIZATION_CODE', 'ITEM', 'TRANSACTION_TYPE_NAME', 'WORK_ORDER_TYPE',
                     'ACTIVITY_ITEM', 'ASSET_NUMBER', 'YEAR_NUMBER', 'TRX_COUNT']]
        data[['ORGANIZATION_CODE', 'ITEM', 'TRANSACTION_TYPE_NAME', 'WORK_ORDER_TYPE', 'ACTIVITY_ITEM', 'ASSET_NUMBER']] = \
            data[['ORGANIZATION_CODE', 'ITEM', 'TRANSACTION_TYPE_NAME', 'WORK_ORDER_TYPE', 'ACTIVITY_ITEM', 'ASSET_NUMBER']].fillna('')
        data['YEAR_NUMBER'] = data['YEAR_NUMBER'].fillna(0)
        data = data.groupby(['ORGANIZATION_CODE', 'ITEM', 'TRANSACTION_TYPE_NAME', 'WORK_ORDER_TYPE',
                             'ACTIVITY_ITEM', 'ASSET_NUMBER', 'YEAR_NUMBER'], as_index=False).sum()
    if program == 'CRM2':
        data['TRX_COUNT'] = 1
        data = data[['ORGANIZATION_CODE', 'ITEM', 'TRANSACTION_TYPE_NAME', 'WORK_ORDER_TYPE',
                     'ASSET_NUMBER', 'YEAR_NUMBER', 'TRX_COUNT']]
        data[['ORGANIZATION_CODE', 'ITEM', 'TRANSACTION_TYPE_NAME', 'WORK_ORDER_TYPE', 'ASSET_NUMBER']] = \
            data[['ORGANIZATION_CODE', 'ITEM', 'TRANSACTION_TYPE_NAME', 'WORK_ORDER_TYPE', 'ASSET_NUMBER']].fillna('')
        data['YEAR_NUMBER'] = data['YEAR_NUMBER'].fillna(0)
        data = data.groupby(['ORGANIZATION_CODE', 'ITEM', 'TRANSACTION_TYPE_NAME', 'WORK_ORDER_TYPE',
                             'ASSET_NUMBER', 'YEAR_NUMBER'], as_index=False).agg({'TRX_COUNT': 'sum'})
    return data
