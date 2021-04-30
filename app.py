import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = pickle.load(open('modelfinal.pkl', 'rb'))

@app.route('/predict-heart-attack', methods=['GET', 'POST'])
def api_pred():
    inputs = request.json
    data_array = []
    result = dict()
    try:
        for input in inputs:
            data_array.append(float(inputs[input]["value"]))

        # if any of the input is undefined return failure.
        if len(data_array) < 7:
            # return faliure result with the provided data.
            result = {item: inputs[item]["value"] for item in inputs}

            result["success"] = False
            result["prediction_result"] = None

            return jsonify(result)

    except:
        result["success"] = False
        result["prediction_result"] = None
        result["message"] = "Invalid Input"

        return jsonify(result)

    data = np.array([data_array])

    prediction = model.predict(data)

    result = {
        'age': inputs['age']['value'],
        'cholestrol': inputs['cholestrol']['value'],
        'systolic_pressure': inputs['systolicPressure']['value'],
        'diastolic_pressure': inputs['diastolicPressure']['value'],
        'bmi': inputs['bmi']['value'],
        'heart_rate': inputs['heartRate']['value'],
        'glucose': inputs['glucoseConc']['value'],
        'prediction_result': bool(prediction[0]),
        'success': True
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
