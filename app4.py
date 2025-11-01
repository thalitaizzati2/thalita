# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.decomposition import PCA

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Data Kepala Sekolah",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI UNTUK MEMUAT DATA ---
@st.cache_data
def load_data():
    """Memuat data dari file CSV."""
    df = pd.read_csv('data_kepsek_dummy.csv')
    return df

# --- FUNGSI UNTUK MENANGANI DATA ---
def clean_and_prepare_data(df):
    """Fungsi untuk menjalankan semua tahap pembersihan data."""
    df_processed = df.copy()
    
    # 1. Simulasi dan penanganan missing values
    # (Dalam aplikasi nyata, ini akan didasarkan pada deteksi NaN yang sebenarnya)
    # Untuk keperluan demo, kita akan mengisi jika ada NaN
    if df_processed.isnull().values.any():
        df_processed.fillna(df_processed.mean(numeric_only=True), inplace=True)
        df_processed.fillna(df_processed.mode().iloc[0], inplace=True) # Untuk kategorikal

    # 2. Koreksi tipe data (contoh, jika Jumlah_Siswa adalah string)
    if df_processed['Jumlah_Siswa'].dtype == 'object':
        df_processed['Jumlah_Siswa'] = pd.to_numeric(df_processed['Jumlah_Siswa'].str.replace(' siswa', ''), errors='coerce')
        df_processed.dropna(subset=['Jumlah_Siswa'], inplace=True) # Hapus jika konversi gagal
        df_processed['Jumlah_Siswa'] = df_processed['Jumlah_Siswa'].astype(int)

    return df_processed

# --- PROSES UTAMA: MUAT DAN BERSIHKAN DATA ---
df_raw = load_data()
df_processed = clean_and_prepare_data(df_raw)

# --- SIDEBAR: FILTER & NAVIGASI ---
st.sidebar.markdown("# 📊 Navigasi & Filter")
page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Dashboard", "Visualisasi", "Analisis Lanjutan", "Data Cleaning"]
)

st.sidebar.markdown("### 🔎 Filter Data")
# Filter untuk kolom kategorikal
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
filtered_df = df_processed.copy()

for col in categorical_cols:
    unique_values = ['Semua'] + list(df_processed[col].unique())
    selected_values = st.sidebar.multiselect(
        f"Filter {col}:",
        options=unique_values,
        default=['Semua']
    )
    if 'Semua' not in selected_values:
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

# Filter untuk kolom numerik
numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols:
    min_val = float(df_processed[col].min())
    max_val = float(df_processed[col].max())
    selected_range = st.sidebar.slider(
        f"Rentang {col}:",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )
    filtered_df = filtered_df[
        (filtered_df[col] >= selected_range[0]) & 
        (filtered_df[col] <= selected_range[1])
    ]

st.sidebar.markdown(f"**Data Tersaring: {len(filtered_df)} baris**")


# --- KONTEN UTAMA BERDASARKAN HALAMAN YANG DIPILIH ---
if page == "Dashboard":
    st.markdown("# 📈 Dashboard Analisis Data")
    st.markdown("Gambaran umum dari data kepala sekolah setelah difilter.")
    
    # Metrik Utama
    st.subheader("Metrik Kunci")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Sekolah", value=len(filtered_df))
    with col2:
        st.metric(label="Rata-rata UN", value=f"{filtered_df['Nilai_UN_RataRata'].mean():.2f}")
    with col3:
        st.metric(label="Rata-rata Usia Kepsek", value=f"{filtered_df['Usia'].mean():.0f} Tahun")
    with col4:
        st.metric(label="Total Siswa", value=f"{filtered_df['Jumlah_Siswa'].sum():,}")

    st.divider()
    
    # Grafik Ringkasan
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Tipe Sekolah")
        # --- PERBAIKAN 1: Mengubah 'Jenis_Sekolah' menjadi 'Tipe_Sekolah' ---
        fig_type = px.pie(
            filtered_df, 
            names='Tipe_Sekolah', 
            title='Proporsi Tipe Sekolah',
            hole=0.4
        )
        st.plotly_chart(fig_type, use_container_width=True)
    
    with col2:
        st.subheader("Distribusi Nilai UN")
        fig_un = px.histogram(
            filtered_df, 
            x="Nilai_UN_RataRata", 
            nbins=20,
            title="Histogram Nilai UN Rata-rata",
            marginal="box"
        )
        st.plotly_chart(fig_un, use_container_width=True)

    st.subheader("Data Tersaring")
    st.dataframe(filtered_df)


elif page == "Visualisasi":
    st.markdown("# 📊 Visualisasi Data Eksploratif")
    st.write("Eksplorasi data lebih dalam melalui berbagai grafik.")
    
    # Pilih kolom untuk visualisasi
    selected_cols_viz = st.multiselect(
        "Pilih kolom numerik untuk divisualisasikan:",
        options=numeric_cols,
        default=['Nilai_UN_RataRata', 'Jumlah_Siswa', 'Usia']
    )
    
    if selected_cols_viz:
        # Histogram
        st.subheader("Histogram Distribusi")
        for col in selected_cols_viz:
            fig = px.histogram(
                filtered_df, 
                x=col, 
                title=f'Histogram {col}',
                nbins=20,
                marginal='box'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Box Plot Perbandingan
        if len(selected_cols_viz) > 1:
            st.subheader("Box Plot Perbandingan")
            fig = px.box(
                filtered_df,
                y=selected_cols_viz,
                title='Box Plot untuk Variabel Numerik'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter Plot
        if len(selected_cols_viz) >= 2:
            st.subheader("Scatter Plot")
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Pilih sumbu X:", selected_cols_viz)
            with col2:
                y_axis = st.selectbox("Pilih sumbu Y:", selected_cols_viz, index=1)
            
            # --- PERBAIKAN 2: Mengubah 'Jenis_Sekolah' menjadi 'Tipe_Sekolah' ---
            fig = px.scatter(
                filtered_df, 
                x=x_axis, 
                y=y_axis,
                color='Tipe_Sekolah',
                title=f'Scatter Plot {y_axis} vs {x_axis}',
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)


elif page == "Analisis Lanjutan":
    st.markdown("# 🔬 Analisis Statistik Lanjutan")
    st.write("Lakukan berbagai analisis statistik untuk menemukan pola dan hubungan dalam data.")
    
    analysis_type = st.selectbox(
        "Pilih jenis analisis:",
        ["Analisis Korelasi", "Uji T", "ANOVA", "Analisis Regresi", "Analisis Klaster", "Analisis PCA (Principal Component Analysis)"]
    )

    if analysis_type == "Analisis Korelasi":
        st.subheader("Analisis Korelasi")
        selected_vars = st.multiselect(
            "Pilih variabel numerik:", numeric_cols, default=numeric_cols[:3]
        )
        if len(selected_vars) >= 2:
            corr_matrix = filtered_df[selected_vars].corr()
            st.write("**Matriks Korelasi:**")
            st.dataframe(corr_matrix.round(2))
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Heatmap Korelasi", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Uji T":
        st.subheader("Uji T (Two-Sample T-Test)")
        var_numerik = st.selectbox("Pilih variabel numerik:", numeric_cols)
        var_kategorikal = st.selectbox("Pilih variabel kategorikal (2 kategori):", categorical_cols)
        if var_numerik and var_kategorikal:
            kategori = filtered_df[var_kategorikal].unique()
            if len(kategori) == 2:
                grup1 = filtered_df[filtered_df[var_kategorikal] == kategori[0]][var_numerik]
                grup2 = filtered_df[filtered_df[var_kategorikal] == kategori[1]][var_numerik]
                t_stat, p_value = stats.ttest_ind(grup1, grup2)
                st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
                if p_value < 0.05:
                    st.success("Ada perbedaan signifikan antara kedua kelompok.")
                else:
                    st.info("Tidak ada perbedaan signifikan antara kedua kelompok.")
                fig = px.box(filtered_df, x=var_kategorikal, y=var_numerik, title=f'Perbandingan {var_numerik}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Variabel kategorikal harus memiliki tepat 2 kategori.")

    elif analysis_type == "ANOVA":
        st.subheader("ANOVA (Analysis of Variance)")
        var_numerik = st.selectbox("Pilih variabel numerik:", numeric_cols, key="anova_num")
        var_kategorikal = st.selectbox("Pilih variabel kategorikal (>2 kategori):", categorical_cols, key="anova_cat")
        if var_numerik and var_kategorikal:
            kategori = filtered_df[var_kategorikal].unique()
            if len(kategori) > 2:
                groups = [filtered_df[filtered_df[var_kategorikal] == cat][var_numerik] for cat in kategori]
                f_stat, p_value = stats.f_oneway(*groups)
                st.write(f"F-statistic: {f_stat:.4f}, P-value: {p_value:.4f}")
                if p_value < 0.05:
                    st.success("Ada perbedaan signifikan antara kelompok-kelompok tersebut.")
                else:
                    st.info("Tidak ada perbedaan signifikan antara kelompok-kelompok tersebut.")
                fig = px.box(filtered_df, x=var_kategorikal, y=var_numerik, title=f'Perbandingan {var_numerik}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Variabel kategorikal harus memiliki lebih dari 2 kategori.")

    elif analysis_type == "Analisis Regresi":
        st.subheader("Analisis Regresi Linear")
        var_dependen = st.selectbox("Pilih variabel dependen (Y):", numeric_cols, key="reg_y")
        var_independen_options = [col for col in numeric_cols if col != var_dependen]
        var_independen = st.multiselect("Pilih variabel independen (X):", var_independen_options, default=var_independen_options[:1])
        if var_dependen and var_independen:
            X = filtered_df[var_independen]
            y = filtered_df[var_dependen]
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            st.write(f"Koefisien Determinasi (R²): {r2:.4f}")
            coef_df = pd.DataFrame({'Variabel': ['Intercept'] + var_independen, 'Koefisien': [model.intercept_] + list(model.coef_)})
            st.dataframe(coef_df)
            if len(var_independen) == 1:
                fig = px.scatter(filtered_df, x=var_independen[0], y=var_dependen, trendline="ols", title=f'Regresi {var_dependen} ~ {var_independen[0]}')
                st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Analisis Klaster":
        st.subheader("Analisis Klaster (K-Means)")
        selected_vars_cluster = st.multiselect("Pilih variabel untuk klaster:", numeric_cols, default=numeric_cols[:2])
        if selected_vars_cluster:
            n_clusters = st.slider("Pilih jumlah klaster:", 2, 10, 3)
            X = filtered_df[selected_vars_cluster]
            X_scaled = StandardScaler().fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_scaled)
            filtered_df['Cluster'] = kmeans.labels_
            
            st.write(f"Distribusi Klaster:")
            st.dataframe(filtered_df['Cluster'].value_counts().sort_index())
            
            if len(selected_vars_cluster) == 2:
                fig = px.scatter(filtered_df, x=selected_vars_cluster[0], y=selected_vars_cluster[1], color='Cluster', title='Visualisasi Klaster')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Visualisasi 2D hanya tersedia untuk 2 variabel yang dipilih.")

    elif analysis_type == "Analisis PCA (Principal Component Analysis)":
        st.subheader("Analisis PCA (Principal Component Analysis)")
        st.write("PCA digunakan untuk mengurangi dimensi data sambil mempertahankan sebanyak mungkin informasi (varians).")
        
        selected_vars_pca = st.multiselect(
            "Pilih variabel numerik untuk PCA:", 
            numeric_cols, 
            default=numeric_cols
        )
        
        if len(selected_vars_pca) >= 2:
            X = filtered_df[selected_vars_pca]
            X_scaled = StandardScaler().fit_transform(X)
            
            # Menentukan jumlah komponen
            n_components = st.slider("Pilih jumlah komponen utama yang akan dianalisis:", 2, len(selected_vars_pca), 2)
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Membuat DataFrame untuk hasil PCA
            pca_cols = [f'PC{i+1}' for i in range(n_components)]
            pca_df = pd.DataFrame(X_pca, columns=pca_cols)
            
            # 1. Scree Plot (Explained Variance)
            st.subheader("1. Scree Plot (Varians yang Dijelaskan)")
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(x=pca_cols, y=explained_variance, name='Varians Individu'))
            fig_scree.add_trace(go.Scatter(x=pca_cols, y=cumulative_variance, name='Varians Kumulatif', mode='lines+markers'))
            fig_scree.update_layout(title='Scree Plot', xaxis_title='Komponen Utama', yaxis_title='Proporsi Varians yang Dijelaskan')
            st.plotly_chart(fig_scree, use_container_width=True)
            
            # 2. Scatter Plot 2D dari PCA
            if n_components >= 2:
                st.subheader("2. Scatter Plot Proyeksi PCA")
                color_col = st.selectbox("Pilih warna berdasarkan:", ['None'] + categorical_cols, key='pca_color')
                
                fig_pca_scatter = px.scatter(
                    pca_df, x='PC1', y='PC2', 
                    color=filtered_df[color_col] if color_col != 'None' else None,
                    title=f'Proyeksi Data pada PC1 ({explained_variance[0]:.2%} Varians) dan PC2 ({explained_variance[1]:.2%} Varians)',
                    labels={'PC1': f'PC1 ({explained_variance[0]:.2%})', 'PC2': f'PC2 ({explained_variance[1]:.2%})'}
                )
                st.plotly_chart(fig_pca_scatter, use_container_width=True)

                # 3. Biplot
                st.subheader("3. Biplot")
                st.write("Biplot menggabungkan scatter plot dengan vektor variabel asli untuk interpretasi.")
                
                # Skala loading vectors
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                
                fig_biplot = px.scatter(
                    pca_df, x='PC1', y='PC2',
                    color=filtered_df[color_col] if color_col != 'None' else None,
                    title='Biplot',
                    labels={'PC1': f'PC1 ({explained_variance[0]:.2%})', 'PC2': f'PC2 ({explained_variance[1]:.2%})'}
                )
                
                for i, var in enumerate(selected_vars_pca):
                    fig_biplot.add_shape(
                        type='line', x0=0, y0=0, x1=loadings[i, 0], y1=loadings[i, 1],
                        line=dict(color="black", width=2)
                    )
                    fig_biplot.add_annotation(
                        x=loadings[i, 0], y=loadings[i, 1],
                        ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y',
                        text=var, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="black"
                    )
                
                st.plotly_chart(fig_biplot, use_container_width=True)


elif page == "Data Cleaning":
    st.markdown("# 🧹 Tahap Pembersihan & Persiapan Data")
    st.write("Halaman ini menunjukkan proses yang telah dilakukan untuk membersihkan data mentah.")
    
    st.header("1. Data Mentah (Raw Data)")
    st.dataframe(df_raw)
    with st.expander("Lihat Info Tipe Data Mentah"):
        buffer = df_raw.info(buf=None)
        st.text(buffer)

    st.divider()
    st.header("2. Data Setelah Pembersihan")
    st.success("Data telah melalui tahap penanganan nilai kosong dan koreksi tipe data.")
    st.dataframe(df_processed)
    with st.expander("Lihat Info Tipe Data Bersih"):
        buffer = df_processed.info(buf=None)
        st.text(buffer)

    st.divider()
    st.header("3. Normalisasi / Standarisasi (Opsional)")
    st.write("Fitur ini dapat diaktifkan jika diperlukan untuk analisis tertentu.")
    if st.checkbox("Lakukan Standarisasi Data (Z-Score)"):
        scaler = StandardScaler()
        df_scaled = df_processed.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        st.dataframe(df_scaled.head())
        csv_scaled = df_scaled.to_csv(index=False).encode('utf-8')
        st.download_button("Unduh Data Standarisasi", data=csv_scaled, file_name='data_standarisasi.csv', mime='text/csv')


# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Unduh Data Tersaring")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Unduh Data Filter (.csv)",
    data=csv,
    file_name='data_kepsek_filter.csv',
    mime='text/csv',
)