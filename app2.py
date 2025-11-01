# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Data Cleaning", layout="wide")
st.title("🧹 Tahap 2: Pembersihan & Persiapan Data")
st.markdown("Program ini akan mendemonstrasikan proses membersihkan data mentah agar siap dianalisis.")

# --- FUNGSI UNTUK MEMUAT DATA ---
@st.cache_data
def load_data():
    """Memuat data dari file CSV."""
    df = pd.read_csv('data_kepsek_dummy.csv')
    return df

# --- PROSES UTAMA ---
# 1. Muat data mentah
df_raw = load_data()

# 2. Buat salinan untuk dimodifikasi
df_processed = df_raw.copy()

st.header("1. Data Mentah (Raw Data)")
st.write("Ini adalah tampilan data yang baru saja kita muat dari file CSV.")
st.dataframe(df_raw)
with st.expander("Lihat Info Tipe Data Mentah"):
    buffer = df_raw.info(buf=None)
    st.text(buffer)

st.divider()

# --- TAHAP 2.A: MENANGANI DATA HILANG (MISSING VALUES) ---
st.header("2. Menangani Data Hilang (Missing Values)")
st.write("Data di dunia nyata seringkali memiliki nilai yang kosong. Kita akan mensimulasikan dan menanganinya.")

# Simulasi: Buat beberapa data kosong secara acak
if st.checkbox("✨ Tambahkan data kosong (NaN) secara acak untuk simulasi"):
    df_processed = df_processed.copy()
    # Acak 5% data di kolom 'Usia' dan 'Lama_Menjabat_Tahun' menjadi kosong
    for col in ['Usia', 'Lama_Menjabat_Tahun']:
        mask = np.random.rand(len(df_processed)) < 0.05
        df_processed.loc[mask, col] = np.nan
    
    st.warning("Data dengan nilai kosong (NaN):")
    st.dataframe(df_processed)

    # Pilih metode untuk menangani missing values
    st.subheader("Pilih Metode Penanganan:")
    handling_method = st.radio(
        "Apa yang ingin Anda lakukan dengan baris yang memiliki data kosong?",
        ("Hapus Baris", "Isi dengan Rata-rata (Mean)", "Isi dengan Nilai Tengah (Median)")
    )

    if handling_method == "Hapus Baris":
        df_processed.dropna(inplace=True)
        st.success("Baris yang mengandung data kosong telah dihapus.")
    elif handling_method == "Isi dengan Rata-rata (Mean)":
        df_processed.fillna(df_processed.mean(numeric_only=True), inplace=True)
        st.success("Data kosong telah diisi dengan nilai rata-rata kolomnya.")
    else: # Median
        df_processed.fillna(df_processed.median(numeric_only=True), inplace=True)
        st.success("Data kosong telah diisi dengan nilai median kolomnya.")

    st.write("Data setelah dibersihkan:")
    st.dataframe(df_processed)

st.divider()

# --- TAHAP 2.B: KOREKSI TIPE DATA ---
st.header("3. Koreksi Tipe Data")
st.write("Terkadang kolom numerik terbaca sebagai teks (string). Kita akan memperbaikinya.")

# Simulasi: Ubah tipe data 'Jumlah_Siswa' menjadi string
df_processed['Jumlah_Siswa'] = df_processed['Jumlah_Siswa'].astype(str) + ' siswa'
st.warning("Kolom 'Jumlah_Siswa' sengaja diubah menjadi tipe teks:")
st.write(df_processed[['Nama_Sekolah', 'Jumlah_Siswa']].head())
st.write(f"Tipe data kolom 'Jumlah_Siswa' sekarang: {df_processed['Jumlah_Siswa'].dtype}")

# Perbaiki tipe data
df_processed['Jumlah_Siswa'] = pd.to_numeric(df_processed['Jumlah_Siswa'].str.replace(' siswa', ''), errors='coerce')
st.success("Kolom 'Jumlah_Siswa' telah dikembalikan ke tipe numerik (integer).")
st.write(f"Tipe data kolom 'Jumlah_Siswa' sekarang: {df_processed['Jumlah_Siswa'].dtype}")
st.write(df_processed[['Nama_Sekolah', 'Jumlah_Siswa']].head())

st.divider()

# --- TAHAP 2.C: NORMALISASI & STANDARISASI ---
st.header("4. Normalisasi & Standarisasi Data Numerik")
st.write("Ini penting untuk algoritma machine learning. Kita akan menyamakan skala data numerik agar tidak ada yang mendominasi.")

# Pilih kolom yang akan diskalakan
numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
selected_cols = st.multiselect(
    "Pilih kolom numerik yang ingin Anda skalakan:",
    options=numeric_cols,
    default=['Usia', 'Nilai_UN_RataRata', 'Jumlah_Siswa']
)

if selected_cols:
    scaling_method = st.radio(
        "Pilih metode penskalaan:",
        ("Normalisasi (Min-Max Scaling)", "Standarisasi (Z-Score Scaling)")
    )

    df_scaled = df_processed.copy()

    if scaling_method == "Normalisasi (Min-Max Scaling)":
        scaler = MinMaxScaler()
        st.info("Normalisasi mengubah data ke dalam rentang 0 hingga 1.")
    else:
        scaler = StandardScaler()
        st.info("Standarisasi mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1.")
    
    df_scaled[selected_cols] = scaler.fit_transform(df_scaled[selected_cols])
    
    st.success("Data telah diskalakan!")
    st.write("Data setelah penskalaan (hanya kolom yang dipilih):")
    st.dataframe(df_scaled[selected_cols].head())

    with st.expander("Lihat Statistik Deskriptif Sebelum dan Sesudah"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sebelum Skala**")
            st.write(df_processed[selected_cols].describe())
        with col2:
            st.write("**Setelah Skala**")
            st.write(df_scaled[selected_cols].describe())
    
    # Update dataframe final
    df_processed = df_scaled

st.divider()

# --- HASIL AKHIR ---
st.header("5. Data Bersih Siap Analisis")
st.success("Proses pembersihan selesai! Data di bawah ini adalah versi final yang siap untuk dianalisis lebih lanjut.")
st.dataframe(df_processed)

# Tombol untuk mengunduh data yang sudah bersih
st.subheader("Unduh Data Bersih")
csv = df_processed.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Unduh sebagai data_kepsek_bersih.csv",
    data=csv,
    file_name='data_kepsek_bersih.csv',
    mime='text/csv',
)
