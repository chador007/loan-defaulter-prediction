from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import joblib
import os
app = Flask(__name__)
model_path = os.path.join(os.path.dirname(__file__), 'stacking_classifier.pkl')
model = joblib.load(model_path,'r')
@app.route("/")
def index():
    return render_template('check.html')
@app.route("/predict", methods = ['POST'])
def prediction():
    if request.method =='POST':
        Loan_amount = float(request.form['Loan_Amount'])
        Funded_amount = float(request.form['Funded_amount'])
        Funded_amount_investor = float(request.form['Funded_Amount_Investor'])  
        Home_ownership = float(request.form['Home_Ownership'])  
        Recieving_balance = float(request.form['Recieving_Balance'])
        Total_recieved_interest = float(request.form['Total_Recieved_Interest'])  
        Recoveries = float(request.form['Recoveries'])
        Total_collection_amount = float(request.form['Total_Collection_Amount'])
        Total_current_balance = float(request.form['Total_Current_Balance'])
        Total_revolving_credit_limit = float(request.form['Total_Revolving_Credit_Limit'])
        x_sample = [[Loan_amount,Funded_amount,Funded_amount_investor,Home_ownership,Recieving_balance,Total_recieved_interest, Recoveries,Total_collection_amount,Total_current_balance,Total_revolving_credit_limit]]
        x = pd.DataFrame(x_sample)
        result = model.predict(x)
        return render_template('output.html', value = result)

if __name__=="__main__":
    app.run(debug = True)

