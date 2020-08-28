import logging
from flask import Flask, jsonify, make_response, request
import pytz
import requests
import pandas as pd
import os.path
import tensorflow as tf
import random
import ConnectDB
import get_input
from datetime import datetime
from threading import Thread

app = Flask(__name__)
tasks_list = dict()
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# run the program according to the program parameters
class Run(Thread):
    def __init__(self, id, program, mode, model, epoch, user, log_file, hidden_units=None, org_dict=None,
                 item_dict=None, activity_dict=None, asset_dict=None):
        Thread.__init__(self)
        self.id = id
        self.program = program
        self.mode = mode
        self.model = model
        self.epoch = epoch
        self.user = user
        self.log_file = log_file
        self.hidden_units = hidden_units
        self.org_dict = org_dict
        self.item_dict = item_dict
        self.activity_dict = activity_dict
        self.asset_dict = asset_dict

    def run(self):
        status = 'In Progress'
        utc_time = datetime.utcnow()
        utc_time_str = utc_time.strftime('%b-%d-%Y %H:%M:%S')
        with open(self.log_file, 'a') as f:
            log_str = '{0} [{1}] Program {2} started by User {3}. Mode: {4}\n'.format(
                utc_time_str, str(self.id), self.program, self.user, self.mode)
            f.write(log_str)
        print(utc_time_str + ' [' + str(self.id) + '] Program started.', flush=True)
        # update task status
        tasks_list[self.id] = {'ID': self.id, 'Program': self.program, 'Mode': self.mode, 'Status': status,
                               'User': self.user, 'Timestamp': utc_time}

        try:
            # read data based on program and mode
            work_data = ConnectDB.read_data(program=self.program, mode=self.mode)
            work_data.columns = map(str.upper, work_data.columns)  # get upper case columns for Oracle
            # group data if it is count prediction
            if self.program in ['PPM2', 'CRM2']:
                work_data = get_input.group_data_by_count(work_data, self.program)
            # if the model file and record exist, read the model details and proceed to different modes
            if os.path.isfile(self.model) and not ConnectDB.read_model_details(self.model).empty:
                # read dict and get mapping; and then generate input data
                # get dict names and input size for the model
                total_features, org_dict_name, item_dict_name, activity_dict_name, asset_dict_name = ConnectDB.read_model_details(
                    self.model, option='dict')
                # read dictionaries
                org_dict, item_dict, activity_dict, asset_dict = get_input.read_dicts(org_dict_name,
                                                                                      item_dict_name,
                                                                                      activity_dict_name,
                                                                                      asset_dict_name)
                # raise error if none of the dict files can be found (possibly model details corrupted)
                if all(name is None for name in [org_dict, item_dict, activity_dict, asset_dict]):
                    utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                    with open(self.log_file, 'a') as f:
                        log_str = '{0} [{1}] Model details exist but could not read valid dicts. ' \
                                  'Please check the table.\n'.format(utc_time_str, str(self.id))
                        f.write(log_str)
                    raise OSError('None of the dict files exist.')
                # get mapping
                work_data_mapped = get_input.get_mapped_df(work_data, org_dict, item_dict, activity_dict,
                                                           asset_dict)
                # get input data
                work_data_x, work_data_y, total_features = get_input.get_input_data(work_data_mapped, org_dict,
                                                                                    item_dict, activity_dict,
                                                                                    asset_dict, self.program,
                                                                                    total_features)
                if work_data_x is None:
                    utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                    with open(self.log_file, 'a') as f:
                        log_str = '{0} [{1}] Error when getting input data.\n'.format(utc_time_str, str(self.id))
                        f.write(log_str)
                    raise ValueError('Error when getting input data.')
                # load the model
                model = tf.keras.models.load_model(self.model)
                # in train mode, train the existing model using input data generated above
                if self.mode == 'TRAIN':
                    # for the training mode, save previous version before training
                    utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                    with open(self.log_file, 'a') as f:
                        log_str = '{0} [{1}] Training started. Loading existing model. Epoch: {3}\n'.format(
                            utc_time_str, str(self.id), self.program, str(self.epoch))
                        f.write(log_str)
                    prev_model_name = str(self.model).replace('.h5', '_prev.h5')
                    model.save(prev_model_name)
                    # start training
                    model.fit(work_data_x, work_data_y, epochs=int(self.epoch))
                    model.save(self.model)
                    utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                    with open(self.log_file, 'a') as f:
                        f.write(utc_time_str + ' [' + str(self.id) + '] Model training complete.\n')
                    print(utc_time_str + ' [' + str(self.id) + '] Model training complete.', flush=True)
                    # update model details in the table
                    model_details = ConnectDB.read_model_details(self.model, option='data')
                    utc_time = datetime.utcnow()
                    model_details['LAST_UPDATED_BY'] = self.user
                    model_details['LAST_UPDATE_DATE'] = utc_time
                    model_details['LAST_UPDATE_PROGRAM'] = self.program
                    ConnectDB.update_model_details(model_details)
                    utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                    with open(self.log_file, 'a') as f:
                        log_str = '{0} [{1}] Model details updated.\n'.format(utc_time_str, str(self.id))
                        f.write(log_str)
                # in test/predict mode, predict the input data generated above using loaded model
                else:
                    predictions = model.predict(work_data_x)
                    if {'TRAIN_FLAG', 'TEST_FLAG', 'NEW_FLAG'}.issubset(work_data.columns):
                        work_data = work_data.drop(columns=['TRAIN_FLAG', 'TEST_FLAG', 'NEW_FLAG'])
                    # get predicted values and calculate differences and confidence levels (vary among programs)
                    if self.program in ['PPM1', 'CRM1']:
                        work_data['PREDICTED_QTY'] = predictions
                        work_data['TRANSACTION_QUANTITY'] *= -1
                        if self.mode == 'TEST':
                            work_data['DIFFERENCE_QTY'] = abs(
                                work_data['TRANSACTION_QUANTITY'] - work_data['PREDICTED_QTY'])
                            work_data['DIFFERENCE_PCT'] = work_data['DIFFERENCE_QTY'] / work_data[
                                'TRANSACTION_QUANTITY']
                            work_data['CONFIDENCE_LEVEL'] = 1 - work_data['DIFFERENCE_PCT']
                            work_data.loc[work_data['CONFIDENCE_LEVEL'] <= 0, ['CONFIDENCE_LEVEL']] = 0
                    else:
                        work_data['PREDICTED_COUNT'] = predictions
                        work_data['PREDICTED_COUNT'] = work_data['PREDICTED_COUNT'].round()
                        if self.mode == 'TEST':
                            work_data['DIFFERENCE_COUNT'] = abs(work_data['TRX_COUNT'] - work_data['PREDICTED_COUNT'])
                            work_data['DIFFERENCE_PCT'] = work_data['DIFFERENCE_COUNT'] / work_data['TRX_COUNT']
                            work_data['CONFIDENCE_LEVEL'] = 1 - work_data['DIFFERENCE_PCT']
                            work_data.loc[work_data['CONFIDENCE_LEVEL'] <= 0, ['CONFIDENCE_LEVEL']] = 0
                    utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                    with open(self.log_file, 'a') as f:
                        if self.mode == 'TEST':
                            # calculate accuracy and add to the log
                            pct_70 = work_data[work_data['CONFIDENCE_LEVEL'] >= 0.7].shape[0] / work_data.shape[0]
                            pct_80 = work_data[work_data['CONFIDENCE_LEVEL'] >= 0.8].shape[0] / work_data.shape[0]
                            pct_90 = work_data[work_data['CONFIDENCE_LEVEL'] >= 0.9].shape[0] / work_data.shape[0]
                            f.write(utc_time_str + ' [' + str(self.id) + '] Test complete.\n')
                            f.write('{0} [{1}] 70% accuracy: {2}, 80% accuracy: {3}, 90% accuracy: {4}\n'.format(
                                utc_time_str, str(self.id), str(pct_70), str(pct_80), str(pct_90)
                            ))
                        else:
                            f.write(utc_time_str + ' [' + str(self.id) + '] Prediction complete.\n')
                    print(utc_time_str + ' [' + str(self.id) + '] Prediction and/or testing complete.', flush=True)
                    # add columns for other information
                    utc_time = datetime.utcnow()
                    work_data['TASK_ID'] = str(self.id)
                    work_data['CREATED_BY'] = self.user
                    work_data['CREATION_DATE'] = utc_time
                    work_data['LAST_UPDATED_BY'] = self.user
                    work_data['LAST_UPDATE_DATE'] = utc_time
                    work_data['LAST_UPDATE_PROGRAM'] = self.program
                    # write work data back to the output table
                    ConnectDB.write_data(self.program, work_data)
                    utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                    with open(self.log_file, 'a') as f:
                        f.write(utc_time_str + ' [' + str(self.id) + '] Output table update complete.\n')
                    print(utc_time_str + ' [' + str(self.id) + '] Output table update complete.', flush=True)
            else:
                # if the model doesn't exist in train mode, a new model will be generated
                # either get mapping directly from work data or from default dicts
                if self.mode == 'TRAIN':
                    utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                    with open(self.log_file, 'a') as f:
                        log_str = '{0} [{1}] Training started. Generating new model. Epoch: {3}\n'.format(
                            utc_time_str, str(self.id), self.program, str(self.epoch))
                        f.write(log_str)
                    # if the dictionaries are specified for the new model (already in memory), use them
                    if all(name is not None for name in [self.org_dict, self.item_dict, self.asset_dict]):
                        org_dict, item_dict, activity_dict, asset_dict = get_input.read_dicts(self.org_dict,
                                                                                              self.item_dict,
                                                                                              self.activity_dict,
                                                                                              self.asset_dict)
                        work_data_mapped = get_input.get_mapped_df(work_data, org_dict, item_dict, activity_dict,
                                                                   asset_dict)
                        work_data_x, work_data_y, total_features = get_input.get_input_data(work_data_mapped, org_dict,
                                                                                            item_dict,
                                                                                            activity_dict, asset_dict,
                                                                                            self.program)
                    # otherwise use the work data to generate dictionaries
                    else:
                        # create dictionaries using work data
                        org_dict, item_dict, activity_dict, asset_dict = get_input.get_dicts(work_data,
                                                                                             program=self.program)
                        # get mapping and generate input data in the same way
                        work_data_mapped = get_input.get_mapped_df(work_data, org_dict, item_dict, activity_dict,
                                                                   asset_dict)
                        work_data_x, work_data_y, total_features = get_input.get_input_data(work_data_mapped, org_dict,
                                                                                            item_dict,
                                                                                            activity_dict,
                                                                                            asset_dict, self.program,
                                                                                            total_features=None)
                        # save dictionaries
                        suffix = str(self.model).replace('.h5', '.txt')
                        self.org_dict = 'org_dict_' + suffix
                        get_input.save_dict(org_dict, self.org_dict)
                        self.item_dict = 'item_dict_' + suffix
                        get_input.save_dict(item_dict, self.item_dict)
                        if activity_dict is not None:
                            self.activity_dict = 'activity_dict_' + suffix
                            get_input.save_dict(activity_dict, self.activity_dict)
                        self.asset_dict = 'asset_dict_' + suffix
                        get_input.save_dict(asset_dict, self.asset_dict)

                    # if the hidden_units parameter exists, use it to construct the model
                    if self.hidden_units is not None:
                        units = str(self.hidden_units).split(';')
                        model = tf.keras.Sequential()
                        units = [int(unit.strip()) for unit in units]
                        # add all layers
                        for unit in units:
                            model.add(tf.keras.layers.Dense(unit, activation='relu'))
                        model.add(tf.keras.layers.Dense(1))
                        model.compile(optimizer='adam', loss='mae')
                        model.fit(work_data_x, work_data_y, epochs=int(self.epoch))
                        model.save(self.model)
                        utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                        with open(self.log_file, 'a') as f:
                            f.write(utc_time_str + ' [' + str(self.id) + '] Model training complete.\n')
                        print(utc_time_str + ' [' + str(self.id) + '] Model training complete.', flush=True)

                        # write to model details table for this new model
                        utc_time = datetime.utcnow()
                        if not ConnectDB.read_model_details(self.model).empty:
                            df = ConnectDB.read_model_details(self.model)
                            model_id = int(df[['MODEL_ID']].squeeze())
                        else:
                            model_id = ConnectDB.get_next_model_id()
                        dic = {'MODEL_ID': [model_id], 'MODEL_FILE_NAME': [self.model],
                               'NUM_FEATURES': [int(total_features)], 'ORG_DICT_NAME': [self.org_dict],
                               'ITEM_DICT_NAME': [self.item_dict],
                               'ACTIVITY_DICT_NAME': [self.activity_dict],
                               'ASSET_DICT_NAME': [self.asset_dict], 'CREATED_BY': [self.user],
                               'CREATION_DATE': [utc_time], 'LAST_UPDATED_BY': [self.user],
                               'LAST_UPDATE_DATE': [utc_time],
                               'LAST_UPDATE_PROGRAM': [self.program]}
                        model_details = pd.DataFrame(dic)
                        ConnectDB.update_model_details(model_details)
                        utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                        with open(self.log_file, 'a') as f:
                            log_str = '{0} [{1}] Model details updated.\n'.format(utc_time_str, str(self.id))
                            f.write(log_str)
                    else:
                        utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                        with open(self.log_file, 'a') as f:
                            log_str = '{0} [{1}] Hidden units not specified for the new model. Program stopped.\n'.format(
                                utc_time_str, str(self.id))
                            f.write(log_str)
                        raise AttributeError('Hidden units not given.')
                # if the model doesn't exist in test/predict mode, an error will be raised
                else:
                    utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                    with open(self.log_file, 'a') as f:
                        log_str = '{0} [{1}] Model file {2} not found. Program stopped.\n'.format(
                            utc_time_str, str(self.id), self.model)
                        f.write(log_str)
                    raise OSError('Model File Not Found.')
            status = 'Completed'

        # catch exceptions and mark the status as error
        except Exception as exc:
            status = 'Error'
            print(exc, flush=True)

        finally:
            tasks_list[self.id] = {'ID': self.id, 'Program': self.program, 'Mode': self.mode, 'Status': status,
                                   'User': self.user, 'Timestamp': utc_time}
            utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
            with open(self.log_file, 'a') as f:
                if status == 'Completed':
                    f.write(utc_time_str + ' [' + str(self.id) + '] Task finished.\n')
                else:
                    f.write(utc_time_str + ' [' + str(self.id) + '] Task error.\n')
            print(utc_time_str + ' [' + str(self.id) + '] Task finished.', flush=True)
            # send back to MRO API
            local_tz = pytz.timezone('Asia/Calcutta')
            ist_time = utc_time.replace(tzinfo=pytz.utc).astimezone(local_tz)
            ist_time_str = ist_time.strftime("%a %b %d %H:%M:%S IST %Y")
            send_back_data = {"operation-id": self.id, "status": status, "program-completion-date": ist_time_str}
            send_back_url = 'http://kasei.hamiltonianusa.com/mroapi/rest/updateProgramStatus'
            send_back_headers = {
                'Content-Type': 'application/json'
            }
            try:
                R = requests.post(send_back_url, json=send_back_data, headers=send_back_headers, timeout=10)
                utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                with open(self.log_file, 'a') as f:
                    f.write(utc_time_str + ' [' + str(self.id) + '] Message sent back to MRO.\n')
                print(utc_time_str + ' [' + str(self.id) + '] Message sent back to MRO.', flush=True)
            except (requests.Timeout, TimeoutError):
                utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                with open(self.log_file, 'a') as f:
                    f.write(utc_time_str + ' [' + str(self.id) + '] Message to MRO timeout, please check connection.\n')
            except requests.RequestException as err:
                utc_time_str = datetime.utcnow().strftime('%b-%d-%Y %H:%M:%S')
                with open(self.log_file, 'a') as f:
                    f.write(utc_time_str + ' [' + str(self.id) + '] Message to MRO error (possibly timeout), please '
                                                                 'check connection.\n')


@app.route('/', methods=['GET'])
def hello():
    indent = '&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp '
    response = '<h2>Welcome</h2>' + \
               'You are now connected to the API.<br>' + \
               '<b>PLEASE READ</b>: Below are some commands you can use:<br>' + \
               '<b>/run</b>: run a task<br>' \
               + indent + 'Usage: invoke program by running \"ML API Call\" in the MRO instance <u>OR</u> ' \
                          'Post a JSON data to http://[current_ip]:[current_port]/run<br>' + \
               indent + 'Parameters:<br>' + \
               indent + '--\'Program\': mandatory, options are: PPM1, PPM2, CRM1, CRM2<br>' + \
               indent + '--\'Mode\': mandatory, options are: TRAIN, TEST, PREDICT<br>' + \
               indent + '--\'Model\': mandatory, name of the model file (.h5)<br>' + \
               indent + '--\'Epoch\': mandatory when training, number of times the training data is trained on the model<br>' + \
               indent + '--\'Hidden_Units\': mandatory when training and the model is not created, structure of the model<br>' + \
               indent + '--\'ID\': optional, six-digit<br>' + \
               indent + '--\'User\': optional, the person who runs the program<br>' + \
               indent + '--\'Log_File\': optional, log file name (.txt) <br>' + \
               indent + '--\'Org_Dict\': optional, dictionary file for orgs (used in training);' \
                        ' if not given when creating a model, this dict file will be generated<br>' + \
               indent + '--\'Item_Dict\': optional, dictionary file for items (used in training);' \
                        ' if not given when creating a model, this dict file will be generated<br>' + \
               indent + '--\'Activity_Dict\': optional, dictionary file for activities (used in training);' \
                        ' if not given when creating a model, this dict file will be generated<br>' + \
               indent + '--\'Asset_Dict\': optional, dictionary file for assets (used in training);' \
                        ' if not given when creating a model, this dict file will be generated<br>' + \
               '<b>/tasks</b>: view all the tasks<br>' \
               + indent + 'Usage: type http://[current_ip]:[current_port]/tasks in the browser ' \
                          '<u>OR</u> curl http://[current_ip]:[current_port]/tasks <br>' \
                          '<b>/tasks/task-id</b>: view the task status using a specific task ID<br>' \
               + indent + 'Usage: type http://[current_ip]:[current_port]/tasks/task-id in the browser ' \
                          '<u>OR</u> curl http://[current_ip]:[current_port]/tasks/task-id <br>' \
                          '<b>/clear</b>: clear all completed or aborted tasks.<br>' \
               + indent + 'Usage: curl -X POST http://[current_ip]:[current_port]/clear<br>' \
                          '<b>/clear/task-id</b>: clear the completed or aborted task with a specific task ID.<br>' \
               + indent + 'Usage: curl -X POST http://[current_ip]:[current_port]/clear/task-id<br>'
    return make_response(response)


# read parameters and invoke the program
@app.route('/run', methods=['POST'])
def invoke_program_with_params():
    json_content = request.get_json(silent=False, force=True)
    if json_content is None:
        return 'JSON reading failed'
    # log could be automatically generated
    utc_time = datetime.utcnow()
    utc_time_str = utc_time.strftime('%Y%m%d_%H%M%S')
    log_file = json_content.get('Log_File', 'log_' + utc_time_str + '.txt')
    # program name
    program_name = json_content.get('Program', None)
    # check program name
    if program_name is None:
        utc_time = datetime.utcnow()
        utc_time_str = utc_time.strftime('%b-%d-%Y %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(utc_time_str + ' Program name missing. Current run aborted.\n')
        print(utc_time_str + ' Program name missing. Current run aborted.', flush=True)
        return make_response('Program name is mandatory!')
    else:
        program_name = program_name.replace('\"', '')
        program_name = program_name.replace('\'', '')
        program_name = program_name.upper()
    if program_name not in ['PPM1', 'PPM2', 'CRM1', 'CRM2']:
        utc_time = datetime.utcnow()
        utc_time_str = utc_time.strftime('%b-%d-%Y %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(utc_time_str + ' Invalid program name. Current run aborted.\n')
        print(utc_time_str + ' Invalid program name. Current run aborted.', flush=True)
        return make_response(program_name + ' is not a valid program name. Please try again.<br>'
                                            'Supported programs are: PPM1, PPM2, CRM1, and CRM2.')
    # check task ID, generate one if not given
    random_id = random.randint(100000, 999999)
    while random_id in tasks_list.keys():
        random_id = random.randint(100000, 999999)
    task_id = json_content.get('ID', random_id)
    task_id = str(task_id)
    if len(task_id) != 6:
        return make_response('Please enter a six-digit ID.')
    if task_id in tasks_list.keys():
        current_status = tasks_list[task_id]['Status']
        if current_status == 'In Progress':
            return make_response('Task ' + str(task_id) + ' is still in progress, please wait', 403)
        print('Task ' + str(task_id) + ' status: ' + current_status + ', restarting...')
    status = 'Started'
    # ML related
    mode = json_content.get('Mode', None)
    epoch = json_content.get('Epoch', None)
    model = json_content.get('Model', None)
    # mode and epoch
    if mode is not None:
        mode = str(mode).strip().upper()
        if mode not in ['TRAIN', 'TEST', 'PREDICT']:
            utc_time = datetime.utcnow()
            utc_time_str = utc_time.strftime('%b-%d-%Y %H:%M:%S')
            with open(log_file, 'a') as f:
                f.write(utc_time_str + ' Invalid mode parameter. Current run aborted.\n')
            print(utc_time_str + ' Invalid mode parameter. Current run aborted.', flush=True)
            return make_response('Invalid action (should be either train, test or predict). Please try again.')
        else:
            # epoch has to be specified in TRAIN mode
            if mode == 'TRAIN' and epoch is None:
                utc_time = datetime.utcnow()
                utc_time_str = utc_time.strftime('%b-%d-%Y %H:%M:%S')
                with open(log_file, 'a') as f:
                    f.write(utc_time_str + ' Epoch missing in train mode. Current run aborted.\n')
                print(utc_time_str + ' Epoch missing in train mode. Current run aborted.', flush=True)
                return make_response('Number of epochs needed in train mode!')
    else:
        utc_time = datetime.utcnow()
        utc_time_str = utc_time.strftime('%b-%d-%Y %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(utc_time_str + ' Mode parameter missing. Current run aborted.\n')
        print(utc_time_str + ' Mode parameter missing. Current run aborted.', flush=True)
        return make_response('Mode parameter missing.')
    # model file name
    if model is not None:
        if not os.path.isfile(model) and mode != 'TRAIN':
            utc_time = datetime.utcnow()
            utc_time_str = utc_time.strftime('%b-%d-%Y %H:%M:%S')
            with open(log_file, 'a') as f:
                log_str = '{0} [{1}] The model file {2} does not exist. Current run aborted.\n'.format(
                    utc_time_str, task_id, model)
                f.write(log_str)
            print(utc_time_str + 'The model file ' + model + ' does not exist. Current run aborted.', flush=True)
            return make_response('The model file ' + model + ' does not exist.')
    else:
        utc_time = datetime.utcnow()
        utc_time_str = utc_time.strftime('%b-%d-%Y %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(utc_time_str + ' Model parameter missing. Current run aborted.\n')
        print(utc_time_str + ' Model parameter missing. Current run aborted.', flush=True)
        return make_response('Model parameter missing.')
    # other parameters
    user = json_content.get('Run_User', 'DEFAULT')
    hidden_units = json_content.get('Hidden_Units', None)
    org_dict = json_content.get('Org_Dict', None)
    item_dict = json_content.get('Item_Dict', None)
    activity_dict = json_content.get('Activity_Dict', None)
    asset_dict = json_content.get('Asset_Dict', None)
    utc_time = datetime.utcnow()
    # invoke the program
    thread = Run(id=task_id, program=program_name, mode=mode, epoch=epoch, model=model, user=user,
                 log_file=log_file, hidden_units=hidden_units, org_dict=org_dict, item_dict=item_dict,
                 activity_dict=activity_dict, asset_dict=asset_dict)
    thread.start()
    dic = {'ID': task_id, 'Program': program_name, 'Mode': mode, 'Status': status, 'User': user,
           'Timestamp': utc_time}
    return make_response(jsonify(dic), 200)


# get task status for one ID
@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task_status(task_id):
    task_id = str(task_id)
    if task_id in tasks_list.keys():
        return make_response(jsonify(tasks_list[task_id]))
    else:
        return make_response('Task ' + str(task_id) + ' not found', 404)


# get task status for all tasks
@app.route('/tasks', methods=['GET'])
def get_tasks():
    if len(tasks_list) == 0:
        return make_response('No tasks created', 404)
    else:
        print(tasks_list, flush=True)
        return make_response(jsonify(tasks_list), 200)


# clear all completed or aborted tasks from the record
@app.route('/clear', methods=['POST'])
def clear_completed_tasks():
    ids_to_del = []
    for task_id in tasks_list.keys():
        if tasks_list[task_id]['Status'] in ['Completed', 'Error']:
            ids_to_del.append(task_id)
    for task_id in ids_to_del:
        print('Task ' + str(task_id) + ' cleared')
        del tasks_list[task_id]
    print('All completed/aborted tasks cleared', flush=True)
    return make_response('All completed/aborted tasks cleared', 200)


# clear one specific ID
@app.route('/clear/<int:task_id>', methods=['POST'])
def clear_task(task_id):
    task_id = str(task_id)
    if task_id not in tasks_list.keys():
        return make_response('Task ' + task_id + ' not found', 404)
    elif tasks_list[task_id]['Status'] not in ['Completed', 'Error']:
        return make_response('Task ' + task_id + ' is still in progress, please wait', 403)
    else:
        del tasks_list[task_id]
        print('Task ' + task_id + ' cleared', flush=True)
        return make_response('Task ' + task_id + ' cleared', 200)


if __name__ == '__main__':
    # run on the specific port in config
    if ConnectDB.config_dict is not None:
        port = int(ConnectDB.config_dict['ML_API_PORT'])
        app.run(host='0.0.0.0', port=port)
    else:
        app.run(host='0.0.0.0')
