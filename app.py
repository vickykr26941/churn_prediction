from flask import Flask, render_template, request
import pickle
import numpy as np

# load the random forest classfier model 
filename = 'churn-prediction-lrc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        cid = int(request.form['customer_id'])
        vintage = int(request.form['vintage'])
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        dependents = int(request.form['dependents'])
        customer_nw_category = int(request.form['customer_nw_category'])
        branch_code = int(request.form['branch_code'])
        days_since_last_transaction = float(request.form['days_since_last_transaction'])
        current_balance = float(request.form['current_balance'])
        average_monthly_balance_prevQ2 = float(request.form['average_monthly_balance_prevQ2'])
        current_month_credit = float(request.form['current_month_credit'])
        previous_month_credit = float(request.form['previous_month_credit'])
        previous_month_debit = float(request.form['previous_month_debit'])


        data = np.array([[cid,vintage,age,gender,dependents,customer_nw_category,branch_code,days_since_last_transaction,current_balance,average_monthly_balance_prevQ2,current_month_credit,previous_month_credit,previous_month_debit]])
        my_prediction = classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
