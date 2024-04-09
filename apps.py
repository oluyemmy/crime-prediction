#!/usr/bin/env python
from flask import Flask, render_template, flash, request, jsonify, Markup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os
import numpy as np
#from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 

# default traveler constants
DEFAULT_AREA = 1
DEFAULT_AGE = 10
DEFAULT_PART = 1
DEFAULT_GENDER = 'Female'
DEFAULT_PREMISES = 101
DEFAULT_DESCENT = 'W'
DEFAULT_STATUS = 'Juv Arrest'


# initializing constant vars
average_crime_rate = 0
# logistic regression modeling
scaler = StandardScaler()
xgb_clf = xgb.XGBClassifier(objective='multi:softmax', 
                            num_class=4,   
                            eval_metric=['merror','mlogloss'], 
                            seed=42)
#lr_model = LogisticRegression()

app = Flask(__name__)


@app.before_first_request
def startup():
    global average_crime_rate, xgb_clf, scaler

    from numpy import genfromtxt
    crime_array = genfromtxt('crime2.csv', delimiter=',')
    average_crime_rate = (np.mean([item[0] for item in crime_array]) * 100)

    X_train, X_test, y_train, y_test = train_test_split([item[1:] for item in crime_array], 
                                                 [item[0] for item in crime_array], test_size=0.5, random_state=42)


    # fit model only once
    #lr_model.fit(X_train, y_train)
    xgb_clf.fit(X_train, y_train)

@app.route("/", methods=['POST', 'GET'])
def submit_new_profile():
    prediction_proba = 0.5
    prediction_actual = 0
    if request.method == 'POST':
        selected_area = request.form['selected_area']
        selected_part = request.form['selected_part']
        selected_age = request.form['selected_age']
        selected_premises = request.form['selected_premises']
        selected_gender = request.form['selected_gender']
        selected_descent = request.form['selected_descent']
        selected_status = request.form['selected_status']

        # assign new variables to live data for prediction
        area = int(selected_area)
        part = int(selected_part)
        age = int(selected_age)
        premises = int(selected_premises)
        
        # point of embarcation
       #### gender
        gender_male = 0
        gender_other = 0
        if (selected_gender=='Male'):
            gender_male = 0
        if (selected_gender=='Others'):
            gender_other = 0
                
            ### age
        descent_h = 0
        descent_o = 0
        descent_w = 0
        descent_x = 0
        if (selected_descent =='H'):
            descent_h = 1
        if (selected_descent == 'O'):
            descent_o = 1
        if (selected_descent == 'W'):
            descent_w = 1
        if (selected_descent == 'X'):
            descent_x = 1
                
        ## Self ethincity 
            
        ### Status
        status_adult_other = 0
        status_invest = 0
        status_juv_other = 0
        status_juv_arrest = 0
        status_unkw = 0
        if (selected_status== 'Adult other'):
            status_adult_other   = 1
        if (selected_status == 'Invest Cont'):
            status_invest = 1
        if (selected_status == 'Juv Arrest'):
            status_juv_arrest = 1
        if (selected_status == 'Juv Other'):
            status_juv_other = 1
        if (selected_status == 'UNK'):
            status_unkw = 1       
 
        # build new array to be in same format as modeled data so we can feed it right into the predictor
        final_feature = [[area,part,age,premises,gender_male,gender_other,
                                descent_h,descent_o,descent_w,descent_x,status_adult_other,
                                status_invest,status_juv_other,status_juv_arrest,status_unkw]]
 

        # add user desinged passenger to predict function
        #Y_pred = lr_model.predict_proba(user_designed_passenger)
        #probability_of_surviving_fictional_character = Y_pred[0][1] * 100

        prediction_proba = xgb_clf.predict_proba(final_feature)
        #prediction_proba = naive_bayes.predict_proba(future_data)
        prediction_proba = np.round(prediction_proba[0][1] * 100,2)
        prediction_actual = xgb_clf.predict(final_feature)

        return render_template('index.html',
            prediction_proba = prediction_proba,
            prediction_actual = prediction_actual,
            selected_part = selected_part,
            selected_gender = selected_gender,
            selected_age = selected_age,
            selected_premises = selected_premises,
            selected_status = selected_status,
            selected_descent = selected_descent,
            selected_area = selected_area)
    else:
        # set default passenger settings
        return render_template('index.html',
            prediction_proba = 0,
            prediction_actual = 0,
            selected_area = DEFAULT_AREA,
            selected_part = DEFAULT_PART,
            selected_age = DEFAULT_AGE,
            selected_premises = DEFAULT_PREMISES,
            selected_gender = DEFAULT_GENDER,
            selected_descent = DEFAULT_DESCENT,
            selected_status = DEFAULT_STATUS)

if __name__=='__main__':
	app.run(debug=False)