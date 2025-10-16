import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_klasifikasi_lemon.joblib")

st.set_page_config(
	page_title="Klasifikasi Lemon",
	page_icon=":lemon:"
)

st.title(":lemon: Klasifikasi Lemon")
st.markdown("Aplikasi machine learning untuk klasifikasi lemon dengan Grade A, Grade B, dan Reject")

diameter = st.slider("Diameter", 45.5, 68.5, 56.6)
berat = st.slider("Berat", 70.0, 145.0, 105.0)
tebal_kulit = st.slider("Tebal Kulit", 3.4, 6.0, 4.3)
kadar_gula = st.slider("Kadar Gula", 6.7, 8.6, 7.7)
asal_daerah = st.pills("Asal Daerah", ["California", "Malang", "Medan"], default="California")
musim_panen = st.pills("Musim Panen", ["Awal","Akhir","Puncak"], default="Awal")
warna = st.pills("Warna", ["Hijau pekat","Kuning kehijauan","Kuning cerah"], default="Hijau pekat")

if st.button("Prediksi", type="primary"):
	data = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,musim_panen,warna]], columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","musim_panen","warna"])
	prediksi = model.predict(data)[0]
	presentase = max(model.predict_proba(data)[0])
	st.success(f"Prediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :lemon: oleh **Raditya Fauzi Pratama**")