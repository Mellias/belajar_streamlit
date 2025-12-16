import streamlit as st
import pandas as pd # untuk menampilkan data
import matplotlib.pyplot as plt # untuk menampilkan grafik
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

st.title("Judul Aplikasi")
st.header("Header Aplikasi")
st.subheader("Subheader Aplikasi")
st.write("Teks")

# Menampilkan data
data = pd.DataFrame({
    'Nama' : ['Tata', 'Titi', 'Tutu'],
    'Umur' : [25, 30, 35]
})
st.write("Ini adalah data yang tampil: ")
st.dataframe(data)

# Menampilkan grafik
plt.plot(['Tata', 'Titi', 'Tutu'], [25, 30, 35])
st.pyplot(plt)

# Membuat user input interaktif
# Nama
user_input = st.text_input('Masukkan nama kamu di sini')
st.caption("Tekan enter kalau sudah selesai menuliskan nama kamu")
if user_input:
    st.text(f"Halo, nama kamu adalah {user_input}")

# Umur
umur = st.slider(f"Umur {user_input} berapa? ", 0, 100, 25)
st.write(f"Umur {user_input} adalah {umur} tahun")

# Tombol untuk mengeksekusi
if st.button("Klik saya!"):
    st.write("Tombol setelah klik")

# Membaca model machine learning
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Interaksi user
st.title("Ini adalah interaksi user untuk prediksi atau klasifikasi")
sepal_length = st.number_input("Sepal length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal width", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal length", min_value=0.0, max_value=10.0, value=1.0)
petal_width = st.number_input("Petal width", min_value=0.0, max_value=10.0, value=0.5)

if st.button("Prediksi sekarang"):
    prediction = model.predict([[
        sepal_length, sepal_width, petal_length, petal_width
    ]])
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.write(f"Ini adalah hasil prediksi: {species [prediction[0]]}")
