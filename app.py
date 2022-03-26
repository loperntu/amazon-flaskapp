from flask import Flask, request, render_template, jsonify
from service_tools.service import Predictor, Preprocessor

app = Flask(__name__)

#initialize Preprocessor class
preprocessor = Preprocessor()
#initialize Predictor class
predictor = Predictor()

# Define API endpoints here:




#==============================


# @app.route('/predict', methods=['GET'])
# def predict():
#     if 'text' in request.args:
#        ......

if __name__ == '__main__':
    # run server
    app.run(host = "140.112.147.112", port = 3000, debug=True)