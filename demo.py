from flask import Flask, render_template, request
import numpy as np
import model

demo = Flask(__name__)

@demo.route("/", methods = ['GET','POST'])

def home_page():
    if request.method == 'POST':
        id = request.form['id']
        season = request.form['season'] 
        mnth = request.form['mnth'] 
        yr = request.form['yr'] 
        hr = request.form['hr'] 
        holiday = request.form['holiday'] 
        weekday = request.form['weekday'] 
        workingday = request.form['workingday'] 
        weathersit = request.form['weathersit'] 
        temp = request.form['temp'] 
        atemp = request.form['atemp'] 
        hum = request.form['hum'] 
        windspeed = request.form['windspeed'] 
        mark = np.array([[id, season, mnth, yr, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed]],dtype=float)
        pred = model.predict(mark)
    
    return render_template("index.html", prediction_text = "Prediction is {}".format(int(pred)))



if __name__ == '__main__':
    demo.run(debug=True)
    
