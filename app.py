from flask import Flask, render_template, request
import numpy as np
import model
import os

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])

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
        prediction_text = "Prediction is {}".format(int(pred))

    return render_template("index.html", **locals())



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    
