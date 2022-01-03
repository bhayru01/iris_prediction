
# %% 0. LIBRARIES
import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd

# %% 1. LOAD MODEL
with open('../models/rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# %% 2. FLASK APP WITH SWAGGER
app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict_single_iris', methods = ['GET'])
def predict_single():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: s_length
        in: query
        type: number
        required: true
      - name: s_width
        in: query
        type: number
        required: true
      - name: p_length
        in: query
        type: number
        required: true
      - name: p_width
        in: query
        type: number
        required: true
    """

    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")

    observation_to_predict = np.array([[s_length,
                                        s_width,
                                        p_length,
                                        p_width]])
    
    model_prediction = str(model.predict(observation_to_predict))

    return model_prediction
    


@app.route('/predict_file_iris', methods = ['POST'])
def predict_batch():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: file_to_predict
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get("file_to_predict"))
    model_predictions = str(list(model.predict(input_data)))

    return model_predictions


# %% 3. RUN APP
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1993)