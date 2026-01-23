# Cara Mengatasi Error dan Menjalankan Sistem

## Masalah yang Terjadi
Error yang Anda alami:
```
Uncaught runtime errors:
ERROR Script error.
```

Ini terjadi karena:
1. **API Backend tidak berjalan** di `http://localhost:8000`
2. **Koneksi gagal** antara Word Add-in dan API

## ✅ Solusi yang Sudah Diterapkan

### 1. Perbaikan Error Handling di `taskpane.js`
- ✅ Menambahkan **global error handler** untuk menangkap semua uncaught errors
- ✅ Menambahkan **unhandled promise rejection handler**
- ✅ Menambahkan fungsi **checkApiConnection()** untuk validasi koneksi API saat startup
- ✅ Menambahkan **timeout handler** (30 detik) untuk request API
- ✅ Menambahkan **try-catch** di semua fungsi async
- ✅ Menampilkan pesan error yang lebih jelas kepada user

### 2. CORS Configuration
- ✅ CORS sudah dikonfigurasi dengan benar di `main.py`

## 🚀 Langkah-langkah Menjalankan Sistem

### Step 1: Jalankan API Backend

Buka terminal baru dan jalankan:

```powershell
# Masuk ke folder project
cd "D:\Documents\Kuliah\semester 7\italic-automation-system"

# Aktifkan virtual environment
.\venv\Scripts\Activate.ps1

# Jalankan API
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Anda akan melihat output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
🚀 Starting Italic Automation API
✅ API ready to serve requests
```

**Test API** dengan membuka browser: http://localhost:8000/docs

---

### Step 2: Rebuild Word Add-in

Buka terminal baru (jangan tutup terminal API):

```powershell
# Masuk ke folder word-addin
cd "D:\Documents\Kuliah\semester 7\italic-automation-system\word-addin"

# Rebuild add-in
npm run build:dev
```

Tunggu hingga build selesai (tidak ada error).

---

### Step 3: Jalankan Word Add-in Dev Server

```powershell
# Di folder word-addin yang sama
npm start
```

Anda akan melihat:
```
webpack compiled successfully
```

---

### Step 4: Buka Microsoft Word

1. Buka Microsoft Word
2. Buka dokumen atau buat dokumen baru
3. Klik tab **Home** → **Show Taskpane** (di ribbon sebelah kanan)
4. Add-in akan terbuka

**Sekarang Anda akan melihat pesan:**
```
✅ API terhubung dan siap
```

Jika API tidak berjalan, Anda akan melihat:
```
❌ Tidak dapat terhubung ke API. Pastikan server berjalan di http://localhost:8000
```

---

## 🔍 Troubleshooting

### Error: "Cannot connect to API"

**Penyebab:** API tidak berjalan

**Solusi:**
```powershell
# Terminal 1: Jalankan API
cd "D:\Documents\Kuliah\semester 7\italic-automation-system"
.\venv\Scripts\Activate.ps1
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Error: "Request timeout"

**Penyebab:** API terlalu lambat atau model belum loaded

**Solusi:**
- Tunggu hingga API selesai loading model (lihat log terminal API)
- Pastikan model ada di folder `models/indobert-italic/final/`

### Error: "CORS policy"

**Penyebab:** CORS tidak dikonfigurasi (sudah diperbaiki)

**Solusi:** CORS sudah dikonfigurasi di `main.py`, rebuild add-in:
```powershell
cd word-addin
npm run build:dev
```

### Error: Build gagal

**Penyebab:** Dependencies belum terinstall

**Solusi:**
```powershell
cd word-addin
npm install
npm run build:dev
```

---

## 📝 Checklist Sebelum Testing

- [ ] ✅ API berjalan di terminal (port 8000)
- [ ] ✅ Word Add-in dev server berjalan (port 3000)
- [ ] ✅ Microsoft Word terbuka
- [ ] ✅ Add-in taskpane terbuka di Word
- [ ] ✅ Status menunjukkan "API terhubung dan siap"

---

## 🧪 Testing

1. **Test Koneksi:**
   - Buka add-in
   - Lihat status: harus "✅ API terhubung dan siap"

2. **Test Deteksi:**
   - Ketik teks di Word: "Sistem ini menggunakan machine learning"
   - Klik "🔍 Deteksi Kata Asing"
   - Harus mendeteksi "machine learning"

3. **Test Format:**
   - Klik "✨ Terapkan Italic + Highlight"
   - Kata "machine learning" harus menjadi italic dan highlighted

---

## 📞 Support

Jika masih ada masalah:
1. Periksa console log di browser (F12)
2. Periksa terminal log API
3. Periksa terminal log Word Add-in dev server
4. Pastikan firewall tidak memblokir port 8000 dan 3000

---

## 🎯 Ringkasan Perubahan

### File yang Dimodifikasi:

1. **`word-addin/src/taskpane/taskpane.js`**
   - ✅ Global error handlers
   - ✅ API connection checker
   - ✅ Timeout handlers
   - ✅ Better error messages
   - ✅ Try-catch blocks

### Yang Tidak Perlu Diubah:

1. **`src/api/main.py`** - CORS sudah OK
2. **`word-addin/webpack.config.js`** - Config sudah OK
3. **Model files** - Model sudah trained

---

**Selamat! Sistem sekarang memiliki error handling yang robust dan user-friendly! 🎉**
