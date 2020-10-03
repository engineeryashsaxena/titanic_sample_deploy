import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from model import * 

app = Flask(__name__)

#training the model when app starts 

df=pd.read_csv('titanic.csv')
Training(df)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age,pclass,sex = [x for x in request.form.values()]
    age=int(age)
    pclass=int(pclass)
    sex=str(sex)
    
    test_data=pd.DataFrame({"Age":[age],
                           "Pclass":[pclass],
                           "Sex":[sex]})
    prediction=Inference(test_data)
    output = prediction['predictions'][0]

    return render_template('index.html', prediction_text='Predicted Label: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)