# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:25:00 2021

@author: Seif Hendawy, Shehab Gamal, Mohamed Hagagy
"""
from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)
model = load('final.pkl')

@app.route('/', methods=["POST", 'GET'])
def home():
    if request.method == "POST":
        dia = [[float(x) for x in request.form.values()]]
        predict = model.predict(dia)
        return render_template('answer.html', predict=predict)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
