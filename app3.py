# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Data Cleaning & Analysis", layout="wide")
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

# --- FITUR FILTER ---
st.header("5. Filter Data")
st.write("Gunakan filter di bawah ini untuk menyaring data sesuai kebutuhan analisis.")

# Filter untuk kolom kategorikal
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    st.subheader("Filter Berdasarkan Kategori")
    
    # Buat filter untuk setiap kolom kategorikal
    for col in categorical_cols:
        if col in df_processed.columns:
            unique_values = df_processed[col].unique()
            selected_values = st.multiselect(
                f"Pilih {col}:",
                options=unique_values,
                default=unique_values
            )
            if selected_values:
                df_processed = df_processed[df_processed[col].isin(selected_values)]

# Filter untuk kolom numerik
numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    st.subheader("Filter Berdasarkan Nilai Numerik")
    
    # Buat filter untuk setiap kolom numerik
    for col in numeric_cols:
        if col in df_processed.columns:
            min_val = float(df_processed[col].min())
            max_val = float(df_processed[col].max())
            
            selected_range = st.slider(
                f"Rentang {col}:",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            
            df_processed = df_processed[
                (df_processed[col] >= selected_range[0]) & 
                (df_processed[col] <= selected_range[1])
            ]

st.write(f"Data setelah difilter: {len(df_processed)} baris")
st.dataframe(df_processed)

st.divider()

# --- STATISTIK DESKRIPTIF ---
st.header("6. Statistik Deskriptif")
st.write("Analisis statistik dasar untuk memahami karakteristik data.")

# Pilih kolom untuk analisis
selected_cols = st.multiselect(
    "Pilih kolom untuk analisis statistik:",
    options=df_processed.columns.tolist(),
    default=df_processed.select_dtypes(include=np.number).columns.tolist()
)

if selected_cols:
    # Statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.dataframe(df_processed[selected_cols].describe())

    # Distribusi frekuensi untuk kolom kategorikal
    categorical_selected = [col for col in selected_cols if df_processed[col].dtype == 'object']
    if categorical_selected:
        st.subheader("Distribusi Frekuensi")
        for col in categorical_selected:
            st.write(f"**Distribusi {col}:**")
            freq_df = df_processed[col].value_counts().reset_index()
            freq_df.columns = [col, 'Frekuensi']
            freq_df['Persentase'] = (freq_df['Frekuensi'] / len(df_processed) * 100).round(2)
            st.dataframe(freq_df)
            
            # Visualisasi distribusi frekuensi
            fig = px.bar(
                freq_df, 
                x=col, 
                y='Frekuensi',
                title=f'Distribusi Frekuensi {col}',
                color='Frekuensi',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Visualisasi deskriptif untuk kolom numerik
    numeric_selected = [col for col in selected_cols if df_processed[col].dtype != 'object']
    if numeric_selected:
        st.subheader("Visualisasi Deskriptif")
        
        # Histogram
        for col in numeric_selected:
            fig = px.histogram(
                df_processed, 
                x=col, 
                title=f'Histogram {col}',
                nbins=20,
                marginal='box'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plot
        if len(numeric_selected) > 1:
            fig = px.box(
                df_processed,
                y=numeric_selected,
                title='Box Plot untuk Variabel Numerik'
            )
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- ANALISIS STATISTIK ---
st.header("7. Analisis Statistik Lanjutan")
st.write("Lakukan berbagai analisis statistik untuk menemukan pola dan hubungan dalam data.")

# Pilih jenis analisis
analysis_type = st.selectbox(
    "Pilih jenis analisis:",
    ["Uji T", "ANOVA", "Analisis Korelasi", "Analisis Regresi", "Analisis Klaster"]
)

if analysis_type == "Uji T":
    st.subheader("Uji T (Two-Sample T-Test)")
    st.write("Uji T digunakan untuk membandingkan rata-rata dua kelompok.")
    
    # Pilih variabel numerik
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    var_numerik = st.selectbox("Pilih variabel numerik:", numeric_cols)
    
    # Pilih variabel kategorikal dengan dua kategori
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    var_kategorikal = st.selectbox("Pilih variabel kategorikal (harus memiliki dua kategori):", categorical_cols)
    
    if var_numerik and var_kategorikal:
        # Periksa jumlah kategori
        kategori = df_processed[var_kategorikal].unique()
        if len(kategori) == 2:
            # Pisahkan data menjadi dua kelompok
            grup1 = df_processed[df_processed[var_kategorikal] == kategori[0]][var_numerik]
            grup2 = df_processed[df_processed[var_kategorikal] == kategori[1]][var_numerik]
            
            # Lakukan uji T
            t_stat, p_value = stats.ttest_ind(grup1, grup2)
            
            # Tampilkan hasil
            st.write(f"**Hasil Uji T:**")
            st.write(f"Variabel Numerik: {var_numerik}")
            st.write(f"Variabel Kategorikal: {var_kategorikal}")
            st.write(f"Kelompok 1 ({kategori[0]}): n={len(grup1)}, mean={grup1.mean():.2f}, std={grup1.std():.2f}")
            st.write(f"Kelompok 2 ({kategori[1]}): n={len(grup2)}, mean={grup2.mean():.2f}, std={grup2.std():.2f}")
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            
            alpha = 0.05
            if p_value < alpha:
                st.write(f"Karena p-value < {alpha}, kita menolak hipotesis nol. Ada perbedaan signifikan antara rata-rata kedua kelompok.")
            else:
                st.write(f"Karena p-value ≥ {alpha}, kita gagal menolak hipotesis nol. Tidak ada perbedaan signifikan antara rata-rata kedua kelompok.")
            
            # Visualisasi
            fig = px.box(
                df_processed, 
                x=var_kategorikal, 
                y=var_numerik,
                title=f'Perbandingan {var_numerik} berdasarkan {var_kategorikal}'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Variabel kategorikal '{var_kategorikal}' harus memiliki tepat dua kategori. Saat ini memiliki {len(kategori)} kategori.")

elif analysis_type == "ANOVA":
    st.subheader("ANOVA (Analysis of Variance)")
    st.write("ANOVA digunakan untuk membandingkan rata-rata lebih dari dua kelompok.")
    
    # Pilih variabel numerik
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    var_numerik = st.selectbox("Pilih variabel numerik:", numeric_cols, key="anova_numeric")
    
    # Pilih variabel kategorikal
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    var_kategorikal = st.selectbox("Pilih variabel kategorikal:", categorical_cols, key="anova_categorical")
    
    if var_numerik and var_kategorikal:
        # Periksa jumlah kategori
        kategori = df_processed[var_kategorikal].unique()
        if len(kategori) > 2:
            # Siapkan data untuk ANOVA
            groups = []
            for cat in kategori:
                groups.append(df_processed[df_processed[var_kategorikal] == cat][var_numerik])
            
            # Lakukan ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Tampilkan hasil
            st.write(f"**Hasil ANOVA:**")
            st.write(f"Variabel Numerik: {var_numerik}")
            st.write(f"Variabel Kategorikal: {var_kategorikal}")
            
            for i, cat in enumerate(kategori):
                st.write(f"Kelompok {i+1} ({cat}): n={len(groups[i])}, mean={groups[i].mean():.2f}, std={groups[i].std():.2f}")
            
            st.write(f"F-statistic: {f_stat:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            
            alpha = 0.05
            if p_value < alpha:
                st.write(f"Karena p-value < {alpha}, kita menolak hipotesis nol. Ada perbedaan signifikan antara rata-rata kelompok-kelompok tersebut.")
            else:
                st.write(f"Karena p-value ≥ {alpha}, kita gagal menolak hipotesis nol. Tidak ada perbedaan signifikan antara rata-rata kelompok-kelompok tersebut.")
            
            # Visualisasi
            fig = px.box(
                df_processed, 
                x=var_kategorikal, 
                y=var_numerik,
                title=f'Perbandingan {var_numerik} berdasarkan {var_kategorikal}'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Variabel kategorikal '{var_kategorikal}' harus memiliki lebih dari dua kategori. Saat ini memiliki {len(kategori)} kategori.")

elif analysis_type == "Analisis Korelasi":
    st.subheader("Analisis Korelasi")
    st.write("Analisis korelasi digunakan untuk mengukur kekuatan dan arah hubungan antara dua variabel numerik.")
    
    # Pilih variabel numerik
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    selected_vars = st.multiselect(
        "Pilih variabel numerik untuk analisis korelasi:",
        numeric_cols,
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    )
    
    if len(selected_vars) >= 2:
        # Hitung matriks korelasi
        corr_matrix = df_processed[selected_vars].corr()
        
        # Tampilkan matriks korelasi
        st.write("**Matriks Korelasi:**")
        st.dataframe(corr_matrix)
        
        # Visualisasi heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Heatmap Matriks Korelasi",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot untuk setiap pasangan variabel
        if len(selected_vars) == 2:
            var1, var2 = selected_vars
            corr_val = corr_matrix.loc[var1, var2]
            
            fig = px.scatter(
                df_processed, 
                x=var1, 
                y=var2,
                title=f'Scatter Plot {var1} vs {var2} (Korelasi: {corr_val:.2f})',
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretasi korelasi
            st.write("**Interpretasi Korelasi:**")
            if abs(corr_val) < 0.2:
                st.write("Korelasi sangat lemah")
            elif abs(corr_val) < 0.4:
                st.write("Korelasi lemah")
            elif abs(corr_val) < 0.6:
                st.write("Korelasi sedang")
            elif abs(corr_val) < 0.8:
                st.write("Korelasi kuat")
            else:
                st.write("Korelasi sangat kuat")
            
            if corr_val > 0:
                st.write("Arah korelasi: Positif (ketika satu variabel naik, variabel lain cenderung naik)")
            else:
                st.write("Arah korelasi: Negatif (ketika satu variabel naik, variabel lain cenderung turun)")
    else:
        st.warning("Pilih setidaknya dua variabel numerik untuk analisis korelasi.")

elif analysis_type == "Analisis Regresi":
    st.subheader("Analisis Regresi Linear")
    st.write("Analisis regresi digunakan untuk memodelkan hubungan antara variabel dependen dan satu atau lebih variabel independen.")
    
    # Pilih variabel dependen dan independen
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    var_dependen = st.selectbox("Pilih variabel dependen (Y):", numeric_cols, key="reg_y")
    var_independen = st.multiselect(
        "Pilih variabel independen (X):",
        [col for col in numeric_cols if col != var_dependen],
        default=[col for col in numeric_cols if col != var_dependen][:1] if len(numeric_cols) > 1 else []
    )
    
    if var_dependen and var_independen:
        # Siapkan data
        X = df_processed[var_independen]
        y = df_processed[var_dependen]
        
        # Buat model regresi
        model = LinearRegression()
        model.fit(X, y)
        
        # Prediksi
        y_pred = model.predict(X)
        
        # Evaluasi model
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Tampilkan hasil
        st.write("**Hasil Analisis Regresi:**")
        st.write(f"Variabel Dependen: {var_dependen}")
        st.write(f"Variabel Independen: {', '.join(var_independen)}")
        st.write(f"Koefisien Determinasi (R²): {r2:.4f}")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        
        # Tampilkan koefisien
        st.write("**Koefisien Regresi:**")
        coef_df = pd.DataFrame({
            'Variabel': ['Intercept'] + var_independen,
            'Koefisien': [model.intercept_] + list(model.coef_)
        })
        st.dataframe(coef_df)
        
        # Visualisasi
        if len(var_independen) == 1:
            # Scatter plot dengan garis regresi
            fig = px.scatter(
                df_processed, 
                x=var_independen[0], 
                y=var_dependen,
                title=f'Regresi Linear: {var_dependen} ~ {var_independen[0]}'
            )
            
            # Tambahkan garis regresi
            fig.add_trace(
                go.Scatter(
                    x=df_processed[var_independen[0]], 
                    y=y_pred,
                    mode='lines',
                    name='Garis Regresi',
                    line=dict(color='red')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Plot residual vs prediksi
            residuals = y - y_pred
            fig = px.scatter(
                x=y_pred, 
                y=residuals,
                title='Plot Residual vs Prediksi',
                labels={'x': 'Nilai Prediksi', 'y': 'Residual'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot prediksi vs aktual
            fig = px.scatter(
                x=y, 
                y=y_pred,
                title='Plot Prediksi vs Aktual',
                labels={'x': 'Nilai Aktual', 'y': 'Nilai Prediksi'}
            )
            
            # Tambahkan garis y=x
            min_val = min(min(y), min(y_pred))
            max_val = max(max(y), max(y_pred))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], 
                    y=[min_val, max_val],
                    mode='lines',
                    name='y = x',
                    line=dict(color='red', dash='dash')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Analisis Klaster":
    st.subheader("Analisis Klaster (K-Means)")
    st.write("Analisis klaster digunakan untuk mengelompokkan data berdasarkan kesamaan karakteristik.")
    
    # Pilih variabel untuk klaster
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    selected_vars = st.multiselect(
        "Pilih variabel untuk analisis klaster:",
        numeric_cols,
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
        key="cluster_vars"
    )
    
    if selected_vars:
        # Pilih jumlah klaster
        n_clusters = st.slider("Pilih jumlah klaster:", min_value=2, max_value=10, value=3)
        
        # Siapkan data
        X = df_processed[selected_vars]
        
        # Normalisasi data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Buat model K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_processed['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Tampilkan hasil
        st.write(f"**Hasil Analisis Klaster (K={n_clusters}):**")
        
        # Tampilkan pusat klaster
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=selected_vars)
        centers_df.index = [f"Cluster {i}" for i in range(n_clusters)]
        st.write("**Pusat Klaster:**")
        st.dataframe(centers_df)
        
        # Tampilkan distribusi klaster
        cluster_counts = df_processed['Cluster'].value_counts().sort_index()
        st.write("**Distribusi Klaster:**")
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Jumlah Data'},
            title='Distribusi Data per Klaster',
            color=cluster_counts.index,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualisasi klaster
        if len(selected_vars) == 2:
            # Scatter plot 2D
            fig = px.scatter(
                df_processed,
                x=selected_vars[0],
                y=selected_vars[1],
                color='Cluster',
                title=f'Visualisasi Klaster: {selected_vars[0]} vs {selected_vars[1]}',
                color_continuous_scale='Rainbow'
            )
            
            # Tambahkan pusat klaster
            fig.add_trace(
                go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode='markers',
                    marker=dict(
                        size=15,
                        symbol='x',
                        color='black'
                    ),
                    name='Pusat Klaster'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        elif len(selected_vars) == 3:
            # Scatter plot 3D
            fig = px.scatter_3d(
                df_processed,
                x=selected_vars[0],
                y=selected_vars[1],
                z=selected_vars[2],
                color='Cluster',
                title=f'Visualisasi Klaster 3D',
                color_continuous_scale='Rainbow'
            )
            
            # Tambahkan pusat klaster
            fig.add_trace(
                go.Scatter3d(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    z=centers[:, 2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        symbol='x',
                        color='black'
                    ),
                    name='Pusat Klaster'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Visualisasi dengan PCA untuk reduksi dimensi
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = df_processed['Cluster'].values
            
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                title='Visualisasi Klaster dengan PCA',
                color_continuous_scale='Rainbow'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"PC1 menjelaskan {pca.explained_variance_ratio_[0]:.2%} varians data")
            st.write(f"PC2 menjelaskan {pca.explained_variance_ratio_[1]:.2%} varians data")
        
        # Analisis karakteristik klaster
        st.write("**Karakteristik Klaster:**")
        for i in range(n_clusters):
            cluster_data = df_processed[df_processed['Cluster'] == i]
            st.write(f"**Cluster {i}:**")
            st.write(f"Jumlah data: {len(cluster_data)}")
            
            # Statistik deskriptif untuk setiap variabel
            for var in selected_vars:
                mean_val = cluster_data[var].mean()
                median_val = cluster_data[var].median()
                std_val = cluster_data[var].std()
                st.write(f"- {var}: Mean={mean_val:.2f}, Median={median_val:.2f}, Std={std_val:.2f}")
            
            st.write("---")

st.divider()

# --- HASIL AKHIR ---
st.header("8. Data Bersih Siap Analisis")
st.success("Proses pembersihan dan analisis selesai! Data di bawah ini adalah versi final yang siap untuk dianalisis lebih lanjut.")
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