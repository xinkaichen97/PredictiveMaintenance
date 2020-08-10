# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:29:43 2019

@author: abhishek.s
"""

# %%
# Improvements- Get string patterns- replace numbers with nx, alphabets with ax, punct with px
# create bow features after data cleaning- replace numbers, punct with spaces/others
# add manual features like Jr, Phd, Dr - help in identifying names
# add data dictonary- for names, statenames, pincodes, city names

# %%Libs
import pandas as pd
import os
import gc
import copy
# import nltk
import numpy as np
# from flashtext import KeywordProcessor
# from pi_nlp import NLP
# from scipy.sparse import csr_matrix, hstack
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.naive_bayes import MultinomialNB
# import matplotlib

# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# import argparse
# from xgboost.sklearn import XGBClassifier
# from time import time
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# import pickle
# import warnings
import cx_Oracle
from sqlalchemy import types, create_engine
import sys
import config


cx_Oracle.init_oracle_client(lib_dir='C:/Oracle/instantclient_19_6')
# cx_Oracle.init_oracle_client(lib_dir='/home/opc/oracle/instantclient_19_8')
sid = cx_Oracle.makedsn('150.136.250.130', 1521, sid='ORCL')
# conn = create_engine('oracle+cx_oracle://MRO:MRO@' + sid)
# con = cx_Oracle.connect(r'MRO/MRO@129.213.36.206:1521/ORCL')
con = cx_Oracle.connect('MRO', 'MRO', sid)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
gc.collect()


# %%def_main
def main():
    df = read_training_data('PPM_QTY_PREDICTION', sheet2=None,
                            col=['PPM_QTY_PREDICTION_ID', 'ORG', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER'])  # )
    print(df)


# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# %%Read & clean data
def read_training_data(sheet1, sheet2, col):
    try:
        query = "SELECT * FROM " + str(sheet1)
        df = pd.read_sql(query, con=con)
    except IOError:
        print('Error: table not found')
        return

    df = df[col]
    # df.dropna(axis=0, how='any', subset=[col], inplace=True)
    # df.drop_duplicates(subset=[col], keep='first', inplace=True)

    if sheet2 is not None:
        try:
            query = "SELECT * FROM " + str(sheet2)
            df2 = pd.read_sql(query, con=con)
        except IOError:
            print('Error: table not found')
            return

        df2.dropna(axis=0, how='any', subset=[col], inplace=True)
        df2.drop_duplicates(subset=[col], keep='first', inplace=True)

        df = pd.concat([df, df2])

        del df2
        gc.collect()

    # df.isnull().sum()
    # df = df.dropna(axis=0, how='any', subset=[col], inplace=False)
    # df[col] = df[col].astype(str)

    return df


def update_data():
    cur = con.cursor()
    statement = 'INSERT INTO PPM_QTY_PREDICTION(PPM_QTY_PREDICTION_ID, ORG, ITEM) values (\'33754\', \'ADH\', \'107549\')'
    cur.execute(statement)
    con.commit()
    print('Update success')


# %%Train_Test_Split
def create_train_dataset(df):
    train = pd.DataFrame()

    for i in df['CATEGORY'].unique():
        temp = df[df['CATEGORY'] == i].sample(n=9990)
        train = pd.concat([train, temp])

    del temp
    gc.collect()
    print('create_train_dataset: train')

    return train


# %%Feature Creation

def create_other_features(data, col):
    # len of words after word tokenization
    data['word_punct'] = [nltk.wordpunct_tokenize(r) for r in data[col]]
    data['word'] = [nltk.word_tokenize(r) for r in data[col]]
    data['len_word_punct'] = data['word_punct'].str.len()
    data['len_word'] = data['word'].str.len()
    data['len_punct'] = data['len_word_punct'] - data['len_word']
    data['len_punct'] = data['len_punct'].clip(lower=0)

    # len of string
    data['len_str'] = data[col].str.len()

    # n_commas
    data['n_commas'] = [r.count(",") for r in data[col]]

    # First word is number?
    data['first_word_numeric'] = data['word_punct'].str[0].str.isnumeric()
    data['first_word_numeric'] = data['first_word_numeric'].replace(np.nan, 0)
    data['first_word_numeric'] = (data['first_word_numeric'] * 1).astype(int)

    # if any number same as len as pincode?
    data['pincode'] = [[s for s in mylist if s.isdigit()] for mylist in data['word']]
    data['pincode'] = [[s for s in mylist if len(str(s)) == 5] for mylist in data['pincode']]
    data['pincode'] = [len(mylist) for mylist in data['pincode']]

    return data[['len_word_punct', 'len_word', 'len_punct', 'len_str', 'n_commas', 'first_word_numeric', 'pincode']]


# BoW Features

bow_vect_default = {'vect': 'tfidf', 'encoding': 'utf-8', 'decode_error': 'strict', 'analyzer': 'word',
                    'ngram_range': (1, 3), 'max_df': 0.5, 'min_df': 0.005, 'max_features': 500, 'lowercase': True,
                    'vocabulary': None, 'binary': True, 'use_idf': False, 'sublinear_tf': False, 'list_stopwords': []}


# Possible inputs-
# vect: 'tfidf','count'
# analyzer: 'word', 'char'
# vocabulary: list or dict

def create_bow_vect(df, textcol, vect_params={}):
    t0 = time()
    conditions = copy.deepcopy(bow_vect_default)
    conditions.update(vect_params)

    # print (conditions)

    if conditions['vect'] == 'count':
        # count vectorizer
        vect = CountVectorizer(encoding=conditions['encoding'], decode_error=conditions['decode_error'],
                               analyzer=conditions['analyzer'],
                               ngram_range=conditions['ngram_range'], max_df=conditions['max_df'],
                               min_df=conditions['min_df'], max_features=conditions['max_features'],
                               vocabulary=conditions['vocabulary'], binary=conditions['binary'],
                               lowercase=conditions['lowercase'], stop_words=conditions['list_stopwords'])

    elif conditions['vect'] == 'tfidf':
        # tfidf vectorizer
        vect = TfidfVectorizer(encoding=conditions['encoding'], decode_error=conditions['decode_error'],
                               analyzer=conditions['analyzer'],
                               ngram_range=conditions['ngram_range'], max_df=conditions['max_df'],
                               min_df=conditions['min_df'], max_features=conditions['max_features'],
                               vocabulary=conditions['vocabulary'], binary=conditions['binary'],
                               use_idf=conditions['use_idf'], sublinear_tf=conditions['sublinear_tf'],
                               lowercase=conditions['lowercase'], stop_words=conditions['list_stopwords'])

    else:
        print('Please input vect as tfidf or count')

    x_train = vect.fit_transform(df[textcol])

    feature_names = vect.get_feature_names()
    # print ('\nFeature Names:\n')
    # print (feature_names)

    t1 = time()

    # print ("\nTotal Time take for bow feature creation: ", round(t1-t0,3), "s")

    return x_train, vect


def create_bow_features(data, col):
    bow_vect = {'vect': 'tfidf', 'encoding': 'utf-8', 'decode_error': 'strict', 'analyzer': 'word',
                'ngram_range': (1, 2), 'max_df': 0.8, 'min_df': 0.001, 'max_features': 1000, 'lowercase': True,
                'vocabulary': None, 'binary': False, 'use_idf': False, 'sublinear_tf': False, 'list_stopwords': []}

    x_train, vect = create_bow_vect(data, col, vect_params=bow_vect)

    # print (x_train.shape)
    # print (vect.get_feature_names())

    bow_vect2 = {'vect': 'tfidf', 'encoding': 'utf-8', 'decode_error': 'strict', 'analyzer': 'char',
                 'ngram_range': (1, 3), 'max_df': 0.8, 'min_df': 0.002, 'max_features': 500, 'lowercase': True,
                 'vocabulary': None, 'binary': False, 'use_idf': False, 'sublinear_tf': False, 'list_stopwords': []}

    x_train2, vect2 = create_bow_vect(data, col, vect_params=bow_vect2)

    # print (x_train2.shape)
    # print (vect2.get_feature_names())
    return x_train, x_train2, vect, vect2


def agg_features(data, col):
    x_train_other = create_other_features(data, col)
    x_train, x_train2, vect, vect2 = create_bow_features(data, col)

    x_train = hstack((x_train, x_train2, x_train_other)).tocsr()

    del x_train2, x_train_other

    return x_train, vect, vect2


def agg_features_score_data(data, col, vect, vect2):
    x_train_other = create_other_features(data, col)
    x_train = vect.transform(data[col])
    x_train2 = vect2.transform(data[col])

    x_train = hstack((x_train, x_train2, x_train_other)).tocsr()
    del x_train2, x_train_other

    return x_train


# %%Model fit

def model_fit_NB(data, train_data, target):
    clf = MultinomialNB(alpha=0.01).fit(train_data, data[target])
    data['pred'] = clf.predict(train_data)
    # df['pred'] = clf.predict(x_df)

    return clf, data


def model_fit_XGB(data, train_data, target):
    XGB = XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=200, silent=True, nthread=-1, gamma=0,
                        subsample=0.8, colsample_bytree=0.6)
    XGB.fit(train_data, data[target])
    data['XGB_pred'] = XGB.predict(train_data)
    # df['XGB_pred']=XGB.predict(x_df)

    return XGB, data


# %%Model_Evaluate

# Confusion Matrix

def model_evaluate(data, target, pred_col):
    labels = list(data[target].unique())
    labels.sort()
    '''
    conf_mat = confusion_matrix(data[target], data[pred_col])
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    '''
    # Classification Report
    print(classification_report(data[target], data[pred_col]))


# %%Train a model from scratch
def train_model(train_sheet1, train_sheet2, train_column_data, target, model_dump_file):
    print('Reading Training Data')
    df = read_training_data(train_sheet1, train_sheet2, train_column_data)
    print('Creating Train Dataset')
    train = create_train_dataset(df)
    print('Creating NLP features')
    x_train, vect, vect2 = agg_features(train, train_column_data)
    print('Fitting Model')
    XGB, data = model_fit_XGB(train, x_train, target)
    print('Model Evauation')
    model_evaluate(data, target, 'XGB_pred')

    print('Dump model to pickle file')
    with open(str(model_dump_file) + '.pkl', 'wb') as fp:
        pickle.dump([XGB, vect, vect2], fp)

    return XGB, vect, vect2


# %%Name_Address to Name & Address

def split_name_address(data, data_col):
    final_col = 'XGB_pred'
    id_col = 'ID'

    df_na = data[data[final_col] == 'Name_Address'][[id_col, data_col]]
    df_na['name'] = df_na[data_col].str.split(",").str[0]

    df_na['address'] = df_na[data_col].str.split(",").str[1:]
    df_na['address'] = df_na['address'].str.join(',')

    data = pd.merge(data, df_na, how='left', on=['ID', data_col])

    return data


# %%Full Name- First Name, Last Name

def read_name_data(file_loc):
    query = "SELECT * FROM " + str(file_loc)
    df_fl = pd.read_sql(query, con=con)

    # df_fl['First Name']=df_fl['First Name'].str.lower()
    df_fl['FIRST_NAME'] = df_fl['FIRST_NAME'].str.replace('.', '')
    # df_fl['Last Name']=df_fl['Last Name'].str.lower()
    df_fl['LAST_NAME'] = df_fl['LAST_NAME'].str.replace('.', '')

    # First Name
    df_f = df_fl.groupby(['FIRST_NAME'], as_index=False).size().reset_index(name='counts')
    df_f['FIRST_NAME'] = df_f['FIRST_NAME'].astype(str)
    dict_f = df_f.set_index('FIRST_NAME').T.to_dict('list')
    list_f = list(df_f['FIRST_NAME'].values)

    # Last_Name
    df_l = df_fl.groupby(['LAST_NAME'], as_index=False).size().reset_index(name='counts')
    df_l['LAST_NAME'] = df_l['LAST_NAME'].astype(str)
    dict_l = df_l.set_index('LAST_NAME').T.to_dict('list')
    list_l = list(df_l['LAST_NAME'].values)

    processor1 = KeywordProcessor()
    processor1.add_keywords_from_list(list_f)
    processor2 = KeywordProcessor()
    processor2.add_keywords_from_list(list_l)

    return dict_f, dict_l, processor1, processor2


def name_split(df, col, dict_f, dict_l, processor1, processor2):
    final_col = 'XGB_pred'

    list_name_type = []
    list_first_name = []
    list_last_name = []
    # Scored Data
    df_n = df[df[final_col] == 'Name'][['ID', col]]
    df_na = df[df[final_col] == 'Name_Address'][['ID', 'name']]
    df_na = df_na.rename(columns={'name': col})
    df_n = pd.concat([df_n, df_na])

    df_n = df_n.reset_index(drop=True)
    # df_n['Data1']=df_n['Data'].str.lower()
    df_n['Data1'] = df_n[col].str.replace('.', '')

    # FirstName_LastName
    # s1.str.contains(' ', case=True, regex=True)
    df_n['name1'] = df_n['Data1'].str.split(" ").str[0]
    df_n['name2'] = df_n['Data1'].str.split(" ").str[1:]
    df_n['name2'] = df_n['name2'].str.join(' ')
    df_n['len_name2'] = df_n['name2'].str.len()

    for index, row in df_n.iterrows():
        document = row['Data1']

        if row['len_name2'] > 1:
            list_name_type.append('firstname_lastname')
            list_first_name.append(row['name1'])
            list_last_name.append(row['name2'])

        else:
            found1 = processor1.extract_keywords(document)

            if len(found1) > 0:
                i = found1[0]
                if i in dict_f:
                    fn_count = dict_f[i][0]
                else:
                    fn_count = 0
            else:
                fn_count = 0

            found2 = processor2.extract_keywords(document)

            if len(found2) > 0:
                i = found2[0]
                if i in dict_l:
                    ln_count = dict_l[i][0]
                else:
                    ln_count = 0
            else:
                ln_count = 0

            if ln_count > fn_count:
                list_name_type.append('lastname')
                list_first_name.append('')
                list_last_name.append(row[col])
            else:
                list_name_type.append('firstname')
                list_first_name.append(row[col])
                list_last_name.append('')

    df_n['name_type'] = list_name_type
    df_n['first_name'] = list_first_name
    df_n['last_name'] = list_last_name

    df = pd.merge(df, df_n[['ID', 'name_type', 'first_name', 'last_name']], how='left', on=['ID'])

    return df


# %%score input file
def score_input_file(file_loc, input_file, col, output_file, model, vect, vect2):
    print('Reading file')
    query = "SELECT * FROM " + str(input_file)
    data = pd.read_sql(query, con=con)
    data[col]

    print('Features Creation')
    x_train = agg_features_score_data(data, col, vect, vect2)
    print('Model Scoring')
    data['XGB_pred'] = model.predict(x_train)
    data['pred_proba'] = np.amax(model.predict_proba(x_train), axis=-1)
    data['XGB_pred'] = data.apply(lambda x: x['XGB_pred'] if x['pred_proba'] >= 0.7 else 'nan', axis=1)
    print('Split Name_Address to Name & Address')
    data = split_name_address(data, col)
    print('Reading Names DB')
    dict_f, dict_l, processor1, processor2 = read_name_data(file_loc)
    print('Split Name to First Name & Last Name')
    data = name_split(data, col, dict_f, dict_l, processor1, processor2)

    print('Writing to output table')
    cols = [x for x in data.columns if x not in (
    ['word_punct', 'word', 'len_word_punct', 'len_word', 'len_punct', 'len_str', 'n_commas', 'first_word_numeric',
     'pincode'])]
    cols2 = ['CATEGORY', 'XGB_pred', 'name', 'address', 'name_type', 'first_name', 'last_name']
    for col in cols2:
        data[col] = data[col].astype(str).str.encode(encoding="utf-8", errors="replace")
    data[cols].to_sql(str(output_file), con, if_exists='append')

    return data[cols]


# %%run_main

if __name__ == '__main__':
    df = read_training_data('PPM_QTY_PREDICTION', sheet2=None,
                            col=['PPM_QTY_PREDICTION_ID', 'ORG', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER'])  # )

    print(df)
    # update_data()
    # df = read_training_data('PPM_QTY_PREDICTION', sheet2=None,
    #                         col=['PPM_QTY_PREDICTION_ID', 'ORG', 'ITEM', 'ACTIVITY_ITEM', 'ASSET_NUMBER']) #)
    # print(df)
