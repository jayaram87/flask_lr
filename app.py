from flask import Flask, render_template, request, redirect, url_for
import json
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            features = np.array([[float(x) for x in request.form.values()]])
            prediction = round(model.predict(features)[0], 2)
            return render_template('prediction.html', price=prediction)
        except:
            return redirect(url_for('home'))

if __name__ == '__main__':
    app.run()