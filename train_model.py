import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Baca data
data = pd.read_csv("dataset_kelulusan.csv")
X = data[['tugas', 'uts', 'uas', 'kehadiran']]
y = data['kelulusan']

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Buat model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "model_kelulusan.pkl")
