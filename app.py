import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model_kelulusan.pkl")

st.title("Prediksi Kelulusan Mahasiswa")

# Input dari pengguna
tugas = st.number_input("Nilai Tugas", 0.0, 100.0, step=1.0)
uts = st.number_input("Nilai UTS", 0.0, 100.0, step=1.0)
uas = st.number_input("Nilai UAS", 0.0, 100.0, step=1.0)
kehadiran = st.number_input("Kehadiran (%)", 0.0, 100.0, step=1.0)

# Prediksi
if st.button("Prediksi"):
    fitur = np.array([[tugas, uts, uas, kehadiran]])
    hasil = model.predict(fitur)[0]
    st.success(f"Hasil Prediksi: {hasil}")
