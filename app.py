import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = pickle.load(open('modelfinal.pkl', 'rb'))

@app.route('/<int:age>/<int:cholestrol>/<int:systolic_pressure>/<int:diastolic_pressure>/<int:bmi>/<int:heart_rate>/<int:glucose>')
def api_pred(age,cholestrol,systolic_pressure,diastolic_pressure,bmi,heart_rate,glucose):
    data=np.array([[int(age),int(cholestrol),int(systolic_pressure),int(diastolic_pressure),int(bmi),int(heart_rate),int(glucose)]])

    prediction=model.predict(data)
	
    result={
            'age':age,
            'cholestrol':cholestrol,
            'systolic_pressure':systolic_pressure,
            'diastolic_pressure':diastolic_pressure,
            'bmi':bmi,
            'heart_rate':heart_rate,
            'glucose':glucose,
            'prediction_result':bool(prediction[0])
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
