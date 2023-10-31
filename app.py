from flask import Flask,render_template,request,jsonify,redirect,url_for
from src.pipeline.prediction_pipeline import CustomData,PredictionPipeline
from src.pipeline.data_storage import MGDB

application = Flask(__name__)
app = application

@app.route('/')
def Home_page():
    return render_template("index.html")

@app.route('/predict_datapoint',methods=['GET','POST'])   
def predict_datapoint():
    if request.method == 'GET':
        return render_template("form.html")
    
    else:
        data = CustomData(
            age = float(request.form.get('age')),
            workclass= str(request.form.get('workclass')),
            education_num = float(request.form.get('education_num')),
            occupation = str(request.form.get('occupation')),
            race = str(request.form.get('race')),
            sex = str(request.form.get('sex')),
            capital_gain = float(request.form.get('capital_gain')),
            capital_loss = float(request.form.get('capital_loss')),
            hours_per_week = float(request.form.get('hours_per_week')),
            country = str(request.form.get('country'))
        )
        
        #This only for make as dataframe to store data into mongo db
        new_data= {
            'age' : float(request.form.get('age')),
            "workclass":str(request.form.get('workclass')),
            'education_num' : float(request.form.get('education_num')),
            'occupation' : str(request.form.get('occupation')),
            'race' : str(request.form.get('race')),
            'sex' : str(request.form.get('sex')),
            "capital_gain" : float(request.form.get('capital_gain')),
            "capital_loss" : float(request.form.get('capital_loss')),
            'hours_per_week' : float(request.form.get('hours_per_week')),
            'country' : str(request.form.get('country'))
        }
        m = MGDB()
        m.Insertion(new_data)
        
        
        final_new_data = data.get_as_dataframe()
        predict_pipline = PredictionPipeline()
        pred = predict_pipline.Predict(final_new_data)
        
        result = pred
        
        return render_template("results.html",final_result = result)



if __name__ == "__main__":
    app.run(host='0.0.0.0')