import json
import requests

URL_PATH = "http://129.213.119.199:5000/run"
HEADERS = {
    'Content-Type': 'application/json'
}

TRAIN_DATA = {
    "Program": "CRM2",
    "Mode": "TRAIN",
    "ID": 100001,
    "Epoch": 1,
    "Hidden_Units": "5000;800",
    "Model": "model_CRM2_five_orgs.h5",
    "Run_User": "098"
    # "Log_File": "log_20200824_163111.txt"
    # "Org_Dict": "org_dict_model_PPM1_five_orgs.txt",
    # "Item_Dict": "item_dict_model_PPM1_five_orgs.txt",
    # "Activity_Dict": "activity_dict_model_PPM1_five_orgs.txt",
    # "Asset_Dict": "asset_dict_model_PPM1_five_orgs.txt"
}

TEST_DATA = {
    "Program": "CRM2",
    "Mode": "TEST",
    "ID": 100001,
    "Model": "model_CRM2_five_orgs.h5",
    "Run_User": "097"
}

NEW_DATA = {
    "Program": "CRM2",
    "Mode": "PREDICT",
    "ID": 100006,
    "Model": "model_CRM2_five_orgs.h5",
    "Run_User": "097"
    # "Log_File": "log_20200823_1341.txt"
}


# R = requests.request("POST", URL_PATH, data=payload, headers=HEADERS)  # suggested by Postman
# R = requests.post(URL_PATH, json=TRAIN_DATA, headers=HEADERS)
# R = requests.post(URL_PATH, json=TEST_DATA, headers=HEADERS)
R = requests.post(URL_PATH, json=NEW_DATA, headers=HEADERS)

if R.ok:
    print(R.request.body)
    print(R.request.url)
    print(R.request.headers)
    print("JSON: ", R.json())
else:
    R.raise_for_status()
