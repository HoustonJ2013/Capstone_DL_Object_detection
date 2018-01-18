from flask import Flask
from flask import render_template, request
import _pickle as pickle
import pandas as pd
import numpy as np
from build_model import TextClassifier, get_data

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', title='Segmantic Segmentation', data=None)


@app.route('/submit', methods=['POST'])
def submit():
    doc = request.form['text1']
    model_select = request.form["model_selected"]
#    if model_select == "Default(resnet34_dilated8 + c1_bilinear)":
#        model_c = model
    if isinstance(doc, basestring):
        doc = [doc]
        #sec_name = model_c.predict(doc)
        #proba = [np.max(model_c.predict_proba(doc))]
        return render_template('index.html', title='Named Entity Recognition', data=0)
                                                                    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
