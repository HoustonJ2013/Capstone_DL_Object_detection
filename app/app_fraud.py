from flask import Flask
from flask import render_template, request
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
import urllib2
from pymongo import MongoClient
from time import sleep
import numpy as np
import pandas as pd
from src.predict import prediction
from src.database import mongobd_insert


app = Flask(__name__)


def read_stream(sleepsec = 2):
    sleep(sleepsec)
    return urllib2.urlopen("http://galvanize-case-study-on-fraud.herokuapp.com/data_point").read()

def connect_db(dbname = "Fraud_prediction",
                   tablename = "Fraud_prediction_table", host="", port= ""):
    client = MongoClient()
    return client


@app.route('/')
def index():
    return render_template('index.html', title='Fraud Prediction: Catch it Before it is Too Late', data=None)


@app.route('/score', methods=['POST'])
def score():
    option = request.form["options"]
    if option == "option2":
        json_inp = read_stream(1)
        json_output = prediction(model, json_inp)
        mongobd_insert(json_output, client, tablename=tablename)
        data_display = zip([json_output[cols_dashboard[0]]],
                           [json_output[cols_dashboard[1]]],
                           [json_output[cols_dashboard[2]]],
                           [json_output[cols_dashboard[3]]])
    elif option == "option1":
        db = client[dbname]
        table = db[tablename]
        cursor = table.find().sort("_id", -1).limit(10)
        col1 = []
        col2 = []
        col3 = []
        col4 = []
        for doc in cursor:
            col1.append(doc[cols_dashboard[0]])
            col2.append(doc[cols_dashboard[1]])
            col3.append(doc[cols_dashboard[2]])
            col4.append(doc[cols_dashboard[3]])
        data_display = zip(col1, col2, col3, col4)
    return render_template('index.html', title='Fraud Prediction: Catch it Before it is Too Late', data=data_display)

if __name__ == '__main__':
    with open("./rf_test.pkl") as f:
        model = pickle.load(f)
    client = connect_db()
    dbname = "Fraud_prediction"
    tablename = "test4"
    cols_dashboard = ["org_name", "name", "payee_name", "risk_level"]
    app.run(host='0.0.0.0', port=8080, debug=True)
