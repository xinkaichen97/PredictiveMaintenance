import pandas as pd
import os
import gc
import cx_Oracle
from sqlalchemy import types, create_engine, Table, MetaData, func
from sqlalchemy.orm import sessionmaker

# read settings from config
config_file_name = 'config.csv'
if os.path.isfile(config_file_name):
    config = pd.read_csv(config_file_name)
    config_dict = {name: value for name, value in zip(config['NAME'], config['VALUE'])}
    cx_Oracle.init_oracle_client(lib_dir=config_dict['ORACLE_CLIENT'])
    sid = cx_Oracle.makedsn(config_dict['DB_HOST'], int(config_dict['DB_PORT']), sid=config_dict['DB_SID'])
    eng = create_engine(
        'oracle+cx_oracle://' + config_dict['DB_USERNAME'] + ':' + config_dict['DB_PASSWORD'] + '@' + sid)
else:
    cx_Oracle.init_oracle_client(lib_dir='/home/opc/oracle/instantclient_19_8')
    sid = cx_Oracle.makedsn('150.136.250.130', 1521, sid='ORCL')
    eng = create_engine('oracle+cx_oracle://MRO:MRO@' + sid)

Session = sessionmaker()
con = eng.connect()
Session.configure(bind=eng)
sess = Session()
meta = MetaData()
gc.collect()


# %%def_main
def main():
    df = read_table('PPM_QTY_PREDICTION',
                            col=['PPM_QTY_PREDICTION_ID', 'ORG', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER'])  # )
    print(df)


# read configuration file
def read_config(config_file_name):
    try:
        config = pd.read_csv(config_file_name)
        config_dict = {name: value for name, value in zip(config['NAME'], config['VALUE'])}
        return config_dict
    except OSError:
        print('Config file not found.')


# read from one table
def read_table(table_name, col=None):
    try:
        query = "SELECT * FROM " + str(table_name)
        df = pd.read_sql(query, con=con)
    except IOError:
        print('Error: table not found')
        return
    if col is not None:
        df = df[col]
    return df


# filter data with certain flag and read
def read_data_with_flag(table_name, flag_name):
    df = pd.DataFrame
    try:
        query = "SELECT * FROM " + str(table_name) + " WHERE " + str(flag_name) + " = 'Y'"
        df = pd.read_sql(query, con=con)
    except IOError:
        print('Error: table not found')
    return df


# read data based on the program and the mode
def read_data(program, mode=None):
    table_name = get_input_table_name(program)
    if mode is not None:
        mode = str(mode).upper()
        flag_name = get_flag_name(mode)
        df = read_data_with_flag(table_name, flag_name)
    else:
        df = read_table(table_name, col=None)
    return df


# read model details with the option of returning only dictionaries
def read_model_details(model_file_name, option=None):
    df = read_table('XXMO_MODEL_DETAILS')
    df.columns = map(str.upper, df.columns)
    df = df[df['MODEL_FILE_NAME'] == model_file_name]
    # if there happens to be more than one record (strongly discouraged), select the most recent one
    if df.shape[0] > 1:
        df.sort('LAST_UPDATED_BY')
        df = df.iloc[[-1], :]
    # if the option is dict, return dictionary files; otherwise return the record with the matched name
    if option == 'dict':
        total_features = int(df['NUM_FEATURES'].squeeze())
        org_dict_name = str(df['ORG_DICT_NAME'].squeeze())
        item_dict_name = str(df['ITEM_DICT_NAME'].squeeze())
        activity_dict_name = str(df['ACTIVITY_DICT_NAME'].squeeze())
        asset_dict_name = str(df['ASSET_DICT_NAME'].squeeze())
        return total_features, org_dict_name, item_dict_name, activity_dict_name, asset_dict_name
    else:
        return df


# program and table names are subject to change
def get_input_table_name(program):
    if program in ['PPM1', 'PPM2']:
        in_table_name = 'XXMO_PPM_IN'
    else:
        in_table_name = 'XXMO_CRM_IN'
    return in_table_name


# return flag name based on mode
def get_flag_name(mode):
    if mode == 'TRAIN':
        flag = 'TRAIN_FLAG'
    elif mode == 'TEST':
        flag = 'TEST_FLAG'
    else:
        flag = 'NEW_FLAG'
    return flag


# return output table name based on the program
def get_output_table_name(program):
    if program == 'PPM1':
        out_table_name = 'XXMO_PPM_OUT_QTY'
    elif program == 'PPM2':
        out_table_name = 'XXMO_PPM_OUT_COUNT'
    elif program == 'CRM1':
        out_table_name = 'XXMO_CRM_OUT_QTY'
    else:
        out_table_name = 'XXMO_CRM_OUT_COUNT'
    return out_table_name


# get new columns based on program and mode
def get_added_columns(program, mode):
    if program in ['PPM1', 'CRM1']:
        if mode == 'PREDICT':
            cols = ['PREDICTED_QTY']
        else:
            cols = ['PREDICTED_QTY', 'DIFFERENCE_QTY', 'DIFFERENCE_PCT', 'CONFIDENCE_LEVEL']
    else:
        if mode == 'PREDICT':
            cols = ['PREDICTED_COUNT']
        else:
            cols = ['PREDICTED_COUNT', 'DIFFERENCE_COUNT', 'DIFFERENCE_PCT', 'CONFIDENCE_LEVEL']
    return cols


# add values of new columns into the table (needs cx_Oracle connection not the sqlalchemy one)
def update_data(program, work_data, mode):
    in_table_name = get_input_table_name(program)
    out_table_name = get_output_table_name(program)
    flag_name = get_flag_name(mode)
    added_cols = get_added_columns(program, mode)

    # copy original work data to output table
    cur = con.cursor()
    query_copy = 'INSERT INTO ' + out_table_name \
                 + 'SELECT * FROM ' + in_table_name \
                 + ' WHERE ' + flag_name + ' = \'Y\''
    cur.execute(query_copy)

    # add predicted columns
    added_data = [tuple(x) for x in work_data[added_cols].values]
    query_update = 'INSERT INTO ' + out_table_name
    work_data.to_sql(out_table_name, con, if_exists='append')
    con.commit()


# write rows into table based on program
def write_data(program, work_data):
    dtyp = {c: types.VARCHAR(work_data[c].str.len().max())
            for c in work_data.columns[work_data.dtypes == 'object'].tolist()}
    out_table_name = get_output_table_name(program)
    work_data.to_sql(out_table_name, con=eng, if_exists='append', index=False, dtype=dtyp)


# write rows into table using table name
def write_table(table_name, data):
    dtyp = {c: types.VARCHAR(data[c].str.len().max())
            for c in data.columns[data.dtypes == 'object'].tolist()}
    data.to_sql(table_name, con=eng, if_exists='append', index=False, dtype=dtyp)


# add data into model details (table name change in here if needed)
def add_model_details(data):
    write_table('XXMO_MODEL_DETAILS', data)


# update model details
def update_model_details(data):
    model_details = Table('XXMO_MODEL_DETAILS'.lower(), meta, autoload=True, autoload_with=eng)
    id = int(data[['MODEL_ID']].squeeze())
    stmt = model_details.delete().where(model_details.c.model_id == id)
    con.execute(stmt)
    write_table('XXMO_MODEL_DETAILS', data)


# get the next model ID used in model details
def get_next_model_id():
    model_details = Table('XXMO_MODEL_DETAILS'.lower(), meta, autoload=True, autoload_with=eng)
    max_id = sess.query(func.max(model_details.c.model_id)).scalar()
    return int(max_id) + 1


if __name__ == '__main__':
    df = read_table('XXMO_MTL_MATERIAL_TRX_TBL', col=None)
    print(df)
