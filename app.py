
from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
       
        age= int(request.form['age'])
        hypertension=int(request.form['hypertension'])
        disease=int(request.form['disease'])
        work=request.form['work']
        glucose= float(request.form['glucose'])
        bmi=float(request.form['bmi'])
        smoking=request.form['smoking']
        gender =request.form['gender']
        married=request.form['married']
        residence=request.form['residence']
        
        
        if work=='Self-employed':
            work=1
        elif work=='Private':
            work=2
        elif work=='Children':
            work=3
        elif work=='Gov_job':
            work=4
        else:
            work==5
            
        if smoking=='never-smoked':
            smoking=1
        elif smoking=='formely-smoked':
            smoking=2
        else:
            smoking=3
    
     
    #gender
    if gender =='Male':
        gender=1
    elif gender=='Female':
       gender=2  
    else:
        gender=3
        

    #married
    if married=='yes':
        married_yes=1
    else:
       married_yes=0
    
   #residence
    if residence=='urban':
        Residence_type_Urban=1
    else:
        Residence_type_Urban=0
    
    details=[age,hypertension,disease,work,glucose,bmi,smoking,gender,married_yes,Residence_type_Urban]
    data_out=np.array(details).reshape(1,-1)
    output=model.predict(data_out)
    output=output.item()
    return render_template('result.html',prediction_text='The chances to have stroke is {}'.format(output))


if __name__ == '__main__':
    app.run(port=5000)