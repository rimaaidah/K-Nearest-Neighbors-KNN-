import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Pastikan folder data ada
if not os.path.exists('data'):
    os.makedirs('data')

# Load dataset tanpa header
df = pd.read_csv('data/Iris.csv', header=None)

# Tetapkan nama kolom manual
df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

# Debug: tampilkan kolom
print("Kolom pada dataset:", df.columns.tolist())

# Fitur dan label
X = df.drop('Species', axis=1)
y = df['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Simpan model & scaler
with open('model_knn.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model dan scaler berhasil disimpan.")
