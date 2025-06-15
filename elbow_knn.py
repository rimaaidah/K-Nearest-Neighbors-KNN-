import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data Iris
df = pd.read_csv('data/Iris.csv')
df.columns = df.columns.str.strip()

# Hapus kolom Id jika ada
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

X = df.drop('Species', axis=1)
y = df['Species']

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Coba nilai k dari 1 sampai 20
k_values = list(range(1, 21))
accuracy_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)

# Plot hasilnya
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('Elbow Method untuk Menentukan Nilai k Terbaik')
plt.xlabel('Jumlah Tetangga (k)')
plt.ylabel('Akurasi')
plt.xticks(k_values)
plt.grid(True)
plt.show()
