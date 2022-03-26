from flask import Flask, request, render_template, jsonify
from service_tools.service import Predictor, Preprocessor

app = Flask(__name__)

#initialize Preprocessor class
preprocessor = Preprocessor()
#initialize Predictor class
predictor = Predictor()

# Define API endpoints here:
@app.route('/')
def home():
    return render_template('index.html')    


@app.route('/predict', methods=['POST'])
def predict():
    # receive ajax payload 
    payload = request.get_json()[0]
    text = payload['text']
    mlmodel = payload['mlmodel']
    #process data 
    doc_vec = preprocessor.get_doc_vec(text)
    sentiment = predictor.predict_sentiment(mlmodel, doc_vec)

    return jsonify(sentiment)



#==============================


# @app.route('/predict', methods=['GET'])
# def predict():
#     if 'text' in request.args:
#        ......

if __name__ == '__main__':
    # run server
    app.run(host = "140.112.147.112", port = 3000, debug=True)