from flask import Flask, jsonify, make_response, request
import json
import time
import random
import ConnectDB
from datetime import datetime, timezone
from threading import Thread

app = Flask(__name__)
tasks_list = dict()


# run the program according to the program name and possibly the task ID
class Run(Thread):
    def __init__(self, id, name, param1, param2):
        Thread.__init__(self)
        self.id = id
        self.name = name
        self.param1 = param1
        self.param2 = param2

    def run(self):
        status = 'In Progress'
        utc_time = datetime.utcnow()
        if self.id in tasks_list.keys():
            tasks_list[self.id]['Status'] = status
            tasks_list[self.id]['Timestamp'] = utc_time
        else:
            tasks_list[self.id] = {'ID': self.id, 'Program': self.name, 'Status': status,
                                   'Timestamp': utc_time}
        try:
            df = []
            if self.name == 'PPM1':
                df = ConnectDB.read_training_data('PPM_QTY_PREDICTION', sheet2=None,
                                                  col=['PPM_QTY_PREDICTION_ID', 'ORG', 'ITEM', 'ACTIVITY_ITEM',
                                                       'ASSET_NUMBER'])
            elif self.name == 'PPM2':
                df = ConnectDB.read_training_data('PPM_QTY_PREDICTION', sheet2=None,
                                                  col=['PPM_QTY_PREDICTION_ID', 'ORG', 'ITEM', 'ACTIVITY_ITEM',
                                                       'ASSET_NUMBER'])
            elif self.name == 'CRM1':
                df = ConnectDB.read_training_data('PPM_QTY_PREDICTION', sheet2=None,
                                                  col=['PPM_QTY_PREDICTION_ID', 'ORG', 'ITEM', 'ACTIVITY_ITEM',
                                                       'ASSET_NUMBER'])
            elif self.name == 'CRM2':
                df = ConnectDB.read_training_data('PPM_QTY_PREDICTION', sheet2=None,
                                                  col=['PPM_QTY_PREDICTION_ID', 'ORG', 'ITEM', 'ACTIVITY_ITEM',
                                                       'ASSET_NUMBER'])
            print(df, flush=True)
            time.sleep(60)  # test purpose only
            status = 'Completed'
        except:
            status = 'Error'
        finally:
            utc_time = datetime.utcnow()
            tasks_list[self.id] = {'ID': self.id, 'Program': self.name, 'Status': status,
                                   'Timestamp': utc_time}
            print('Task ' + str(self.id) + ' finished', flush=True)
            # return make_response(jsonify(tasks_list[self.id]))


@app.route('/', methods=['GET'])
def hello():
    indent = '&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp '
    response = '<h2>Welcome</h2>' + \
               'You are now connected to the API.<br>' + \
               '<b>PLEASE READ</b>: Below are some commands you can use:<br>' + \
               '/run?program=program-name[&amp;id=task-id][&amp;param1=p1][&amp;param2=p2][&amp;startDate=date1]: start a task<br>' + \
               indent + 'Program name: mandatory, options are: PPM1, PPM2, CRM1, CRM2<br>' + \
               indent + 'Task ID: optional, six-digit<br>' + \
               indent + 'Parameter1: optional, details TBD<br>' + \
               indent + 'Parameter2: optional, details TBD<br>' + \
               indent + 'Start Date: optional, format: dd-mmm-yy (e.g. 01-AUG-20)<br>' + \
               '/tasks: view all the tasks<br>' + \
               '/tasks/task-id: view the task status using a specific task ID<br>' + \
               '/clear: clear all completed or aborted tasks<br>' + \
               '/clear/task-id: clear the completed or aborted task with a specific task ID<br>'
    return make_response(response)


# CHOOSE PROGRAM NAME (MANDATORY, STRING OR NUM), TASK ID (OPTIONAL) AND OTHER PARAMS (OPTIONAL)
@app.route('/run', methods=['GET'])
def invoke_program_with_params():
    # program name
    program_name = request.args.get('program')
    if program_name is None:
        return make_response('Program name is mandatory!')
    else:
        program_name = program_name.replace('\"', '')
        program_name = program_name.replace('\'', '')
        program_name = program_name.upper()
    if program_name not in ['PPM1', 'PPM2', 'CRM1', 'CRM2']:
        return make_response(program_name + ' is not a valid program name. Please try again.<br>'
                                            'Supported programs are: PPM1, PPM2, CRM1, and CRM2.')
    # task ID
    random_id = random.randint(100000, 999999)
    while random_id in tasks_list.keys():
        random_id = random.randint(100000, 999999)
    task_id = request.args.get('id', random_id)
    task_id = str(task_id)
    if len(task_id) != 6:
        return make_response('Please enter a six-digit ID')
    if task_id in tasks_list.keys():
        current_status = tasks_list[task_id]['Status']
        if current_status == 'In Progress':
            return make_response('Task ' + str(task_id) + ' is still in progress, please wait', 403)
        print('Task ' + str(task_id) + ' status: ' + current_status + ', restarting...')
    else:
        print('Task ' + str(task_id) + ' started', flush=True)
    status = 'Started'
    # other parameters (TBD)
    param1 = request.args.get('param1')
    param2 = request.args.get('param2')
    start_date_str = request.args.get('startDate')
    if start_date_str is not None:
        try:
            start_date = datetime.strptime(start_date_str, '%d-%b-%y')
        except ValueError:
            return make_response('Date format not supported. Please use dd-mmm-yy as in 01-AUG-20.')
        print(start_date, flush=True)
    utc_time = datetime.utcnow()
    thread = Run(id=task_id, name=program_name, param1=param1, param2=param2)
    thread.start()
    dic = {'ID': task_id, 'Program': program_name, 'Status': status,
           'Timestamp': utc_time}
    return make_response(jsonify(dic), 200)


# @app.route('/run', methods=['GET'])
# def invoke_program_new():
#     id = random.randint(100000, 999999)
#     while id in tasks_list.keys():
#         id = random.randint(100000, 999999)
#     print('Generating new ID:' + str(id), flush=True)
#     status = 'Started'
#     print('Task ' + str(id) + ' started', flush=True)
#     utc_time = datetime.utcnow()
#     thread = Run(id=id, name='PPM')
#     thread.start()
#     return make_response(jsonify({'ID': id, 'Status': status, 'Timestamp': utc_time}), 200)


# @app.route('/run/<int:task_id>', methods=['GET'])
# def invoke_program_with_id(task_id):
#     if task_id in tasks_list.keys():
#         current_status = tasks_list[task_id]['Status']
#         if current_status == 'In Progress':
#             return make_response('Task ' + str(task_id) + ' is still in progress, please wait', 403)
#         print('Task ' + str(task_id) + ' status: ' + current_status + ', restarting...')
#     else:
#         print('Generating new ID:' + str(task_id), flush=True)
#         print('Task ' + str(task_id) + ' started', flush=True)
#     status = 'Started'
#     utc_time = datetime.utcnow()
#     thread = Run(id=task_id, name='PPM')
#     thread.start()

#     return make_response(jsonify({'ID': task_id, 'Status': status, 'Timestamp': utc_time}), 200)


@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task_status(task_id):
    task_id = str(task_id)
    if task_id in tasks_list.keys():
        status = tasks_list[task_id]['Status']
        program_name = tasks_list[task_id]['Program']
        utc_time = datetime.utcnow()
        return make_response(jsonify({'ID': task_id, 'Program': program_name, 'Status': status, 'Timestamp': utc_time}), 200)
    else:
        return make_response('Task ' + str(task_id) + ' not found', 404)


@app.route('/tasks', methods=['GET'])
def get_tasks():
    if len(tasks_list) == 0:
        return make_response('No tasks created', 404)
    else:
        print(tasks_list, flush=True)
        return make_response(jsonify(tasks_list), 200)


@app.route('/clear', methods=['GET'])
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


@app.route('/clear/<int:task_id>', methods=['GET'])
def clear_task(task_id):
    if task_id not in tasks_list.keys():
        return make_response('Task ' + str(task_id) + ' not found', 404)
    elif tasks_list[task_id]['Status'] not in ['Completed', 'Error']:
        return make_response('Task ' + str(task_id) + ' is still in progress, please wait', 403)
    else:
        del tasks_list[task_id]
        print('Task ' + str(task_id) + ' cleared', flush=True)
        return make_response('Task ' + str(task_id) + ' cleared', 200)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
