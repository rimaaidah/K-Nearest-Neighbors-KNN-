from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Buat folder jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load model dan scaler
with open('model_knn.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "File tidak ditemukan."

    file = request.files['file']
    if file.filename == '':
        return "Tidak ada file yang dipilih."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # ==============================
        # Baca dan Bersihkan Data
        # ==============================
        df = pd.read_csv(filepath)

        if 'Species' in df.columns:
            df = df.drop('Species', axis=1)

        if df.shape[1] > 4:
            df = pd.read_csv(filepath, header=None)

        df = df.iloc[:, :4]
        df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

        # ==============================
        # Prediksi
        # ==============================
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)

        df['Predicted_Species'] = predictions
        hasil = df.to_dict(orient='records')

        # =======================================
        # Elbow Method (tanpa cross-validation)
        # =======================================
        elbow_scores = []
        k_values = list(range(1, 21))

        for k in k_values:
            model_k = KNeighborsClassifier(n_neighbors=k)
            model_k.fit(X_scaled, predictions)  # Fit dengan label prediksi dummy
            y_pred = model_k.predict(X_scaled)
            acc = accuracy_score(predictions, y_pred)
            elbow_scores.append(acc)

        # Simpan grafik Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, elbow_scores, marker='o')
        plt.title('Elbow Method (tanpa CV)')
        plt.xlabel('Jumlah Tetangga (k)')
        plt.ylabel('Akurasi')
        plt.grid(True)
        plt.savefig('static/elbow.png')
        plt.close()

        # =======================================
        # Cross-Validation Elbow Method (CV=5)
        # =======================================
        cv_scores = []

        for k in k_values:
            model_k = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(model_k, X_scaled, predictions, cv=5)
            cv_scores.append(scores.mean())

        # Simpan grafik CV Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, cv_scores, marker='o', color='green')
        plt.title('Elbow Method dengan Cross-Validation (5-Fold)')
        plt.xlabel('Jumlah Tetangga (k)')
        plt.ylabel('Rata-rata Akurasi CV')
        plt.grid(True)
        plt.savefig('static/elbow_cv.png')
        plt.close()

        return render_template('result.html', hasil=hasil)

    except Exception as e:
        return f"Terjadi error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
