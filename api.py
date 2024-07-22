from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

# Flask uygulamasını başlatma
app = Flask(__name__)

# CORS ayarları
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)

# Model dosya yolu
model_path = 'best_model.pkl'

# Modeli ve label_encoder'ı yükleme (pickle dosyasından)
def load_model_and_encoder(model_path):
    with open(model_path, 'rb') as file:
        model, label_encoder = pickle.load(file)
    return model, label_encoder

# API anahtarını kontrol eden fonksiyon
def check_api_key():
    api_key = request.headers.get('API-Key')
    # Gerçek bir API anahtarını burada kontrol edin
    if api_key == 'apikey':  # Örnek anahtar, gerçek anahtarınızı buraya yazın
        return True
    return False

# /predict endpoint'i
@app.route('/predict', methods=['POST'])
def predict_churn():
    if request.method == 'POST' and request.is_json:
        if not check_api_key():
            return jsonify({'error': 'Unauthorized access'}), 401

        input_data = request.json  # JSON formatındaki veriyi al
        recency = float(input_data['recency'])
        frequency = float(input_data['frequency'])
        monetary = float(input_data['monetary'])

        features = pd.DataFrame([[recency, frequency, monetary]], columns=['recency', 'frequency', 'monetary'])

        # Modeli ve label_encoder'ı yükle
        model, label_encoder = load_model_and_encoder(model_path)

        # Tahmin yap
        prediction = model.predict(features)[0]
        segment = label_encoder.inverse_transform([prediction])[0]

        return jsonify({'SegmentPrediction': segment}), 200
    else:
        return jsonify({'error': 'Only POST requests with JSON data are allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
