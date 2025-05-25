from flask import Flask, render_template, request
import pickle
import subprocess


app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define route to render the form

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Route for the registration page
@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

@app.route('/CausesSymptoms')
def CausesSymptoms():
    return render_template('CausesSymptoms.html')

@app.route('/skinhair')
def skinhair():
    return render_template('skinhair.html')

@app.route('/meditation')
def meditation():
    return render_template('meditation.html')

@app.route('/remedies')
def remedies():
    return render_template('remedies.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/registration-success')
def registration_success():
    return render_template('registration_success.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/lifestyle')
def lifestyle():
    return render_template('lifestyle.html')

# Define route to handle form submission and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = int(request.form['age'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    bmi = float(request.form['bmi'])
    blood_group = int(request.form['blood_group'])
    pulse_rate = int(request.form['pulse_rate'])
    cycle = int(request.form['cycle'])
    cycle_length = int(request.form['cycle_length'])
    marriage_status = float(request.form['marriage_status'])
    pregnant = int(request.form['pregnant'])
    abortions = int(request.form['abortions'])
    hip = int(request.form['hip'])
    waist = int(request.form['waist'])
    waist_hip_ratio = float(request.form['waist_hip_ratio'])
    weight_gain = int(request.form['weight_gain'])
    hair_growth = int(request.form['hair_growth'])
    skin_darkening = int(request.form['skin_darkening'])
    hair_loss = int(request.form['hair_loss'])
    pimples = int(request.form['pimples'])
    fast_food = float(request.form['fast_food'])
    reg_exercise = int(request.form['reg_exercise'])
    
    # Preprocess the input data if necessary
    # For example, convert strings to numerical values
    
    # Make prediction using the model
    prediction = model.predict([[age, weight, height, bmi, blood_group, pulse_rate, cycle, cycle_length,
                                 marriage_status, pregnant, abortions, hip, waist, waist_hip_ratio,
                                 weight_gain, hair_growth, skin_darkening, hair_loss, pimples,
                                 fast_food, reg_exercise]])
    
    # Determine the prediction result
    if prediction == 1:
        result = "PCOS"
    else:
        result = "No PCOS"
    
    # Render the template with the prediction result
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
