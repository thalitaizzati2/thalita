import pandas as pd
import random
import numpy as np

# --- Konfigurasi ---
# Untuk memastikan data yang dihasilkan selalu sama setiap kali dijalankan
random.seed(42)
np.random.seed(42)

# --- Fungsi Pembantu ---
def generate_nama():
    """Menghasilkan nama Indonesia acak."""
    nama_depan = ['Ahmad', 'Siti', 'Budi', 'Dewi', 'Agus', 'Rina', 'Eko', 'Indah', 'Fajar', 'Maya', 'Rizki', 'Sari', 'Hendra', 'Fitri', 'Bayu', 'Wati']
    nama_belakang = ['Susanto', 'Wulandari', 'Pratama', 'Saputra', 'Hidayat', 'Permata', 'Pangestu', 'Kusuma', 'Nugroho', 'Siregar', 'Hakim', 'Putri', 'Laksono', 'Rahayu']
    return f"{random.choice(nama_depan)} {random.choice(nama_belakang)}"

def generate_nama_sekolah(tipe, lokasi):
    """Menghasilkan nama sekolah berdasarkan tipe dan lokasi."""
    if tipe == 'Negeri':
        return f"SMAN {random.randint(1, 20)} {lokasi}"
    else:
        adj = ['Budi Mulia', 'Cahaya', 'Mandiri', 'Harapan', 'Global', 'Nusantara']
        noun = ['Bangsa', 'Ilmu', 'Karya', 'Jaya', 'Pertiwi']
        return f"SMK Swasta {random.choice(adj)} {random.choice(noun)} {lokasi}"

# --- List Opsi untuk Data Kategorikal ---
lokasi_list = ['Jakarta', 'Bandung', 'Surabaya', 'Yogyakarta', 'Medan']
pendidikan_list = ['S1', 'S2', 'S3']
gayakepemimpinan_list = ['Transformasional', 'Transaksional', 'Laissez-faire']

# --- Proses Pembuatan Data ---
data = []
for i in range(100):
    # 1. Data Demografis
    id_kepsek = i + 1
    nama = generate_nama()
    jenis_kelamin = random.choice(['Laki-laki', 'Perempuan'])
    usia = random.randint(40, 62)

    # 2. Data Profesional
    # Probabilitas lebih tinggi untuk S2
    pendidikan_terakhir = random.choices(pendidikan_list, weights=[0.2, 0.6, 0.2])[0]
    lama_menjabat = random.randint(1, 18)
    lama_mengajar = max(0, (usia - 22) - random.randint(0, 10)) # Asumsi mulai mengajar umur 22

    # 3. Data Konteks Sekolah
    tipe_sekolah = random.choice(['Negeri', 'Swasta'])
    lokasi = random.choice(lokasi_list)
    nama_sekolah = generate_nama_sekolah(tipe_sekolah, lokasi)
    # Jumlah siswa lebih tinggi di kota besar dan sekolah negeri
    if lokasi == 'Jakarta' or lokasi == 'Surabaya':
        jumlah_siswa = random.randint(800, 1500)
    else:
        jumlah_siswa = random.randint(300, 900)
    if tipe_sekolah == 'Swasta':
        jumlah_siswa = int(jumlah_siswa * random.uniform(0.6, 0.9))

    # 4. Data Kinerja & Kepemimpinan
    gaya_kepemimpinan = random.choice(gayakepemimpinan_list)
    tingkat_kelulusan = round(random.uniform(92.0, 100.0), 2)

    # 5. Variabel Utama (Nilai UN) dengan sedikit korelasi
    # Skor dasar + noise
    nilai_un = 75 + np.random.normal(0, 5)
    
    # Tambahkan faktor pengaruh
    if pendidikan_terakhir == 'S3':
        nilai_un += 2.5
    if gaya_kepemimpinan == 'Transformasional':
        nilai_un += 1.5
    if lokasi == 'Yogyakarta': # Diasumsikan sebagai kota pendidikan
        nilai_un += 1.0
    
    # Batasi nilai agar realistis
    nilai_un = max(60, min(100, round(nilai_un, 2)))

    # Masukkan ke dictionary
    data.append({
        'ID_Kepsek': id_kepsek,
        'Nama': nama,
        'Jenis_Kelamin': jenis_kelamin,
        'Usia': usia,
        'Pendidikan_Terakhir': pendidikan_terakhir,
        'Lama_Mengajar_Tahun': lama_mengajar,
        'Lama_Menjabat_Tahun': lama_menjabat,
        'Nama_Sekolah': nama_sekolah,
        'Tipe_Sekolah': tipe_sekolah,
        'Lokasi': lokasi,
        'Jumlah_Siswa': jumlah_siswa,
        'Gaya_Kepemimpinan': gaya_kepemimpinan,
        'Tingkat_Kelulusan_Persen': tingkat_kelulusan,
        'Nilai_UN_RataRata': nilai_un
    })

# --- Buat DataFrame dan Simpan ke CSV ---
df = pd.DataFrame(data)

# Simpan ke file CSV
df.to_csv('data_kepsek_dummy.csv', index=False)

print("✅ File 'data_kepsek_dummy.csv' berhasil dibuat dengan 100 baris data.")
print("\n--- 5 Baris Pertama Data ---")
print(df.head())