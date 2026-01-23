# 🔧 PANDUAN MEMPERBAIKI ERROR DI BRANCH ME2

## 🔍 MASALAH YANG DITEMUKAN

### 1. **Python Dependencies Tidak Terinstall (CRITICAL)**
Branch `me2` tidak memiliki dependencies Python yang dibutuhkan untuk menjalankan API backend.

**Error yang terlihat:**
```
ModuleNotFoundError: No module named 'fastapi'
ModuleNotFoundError: No module named 'torch'
```

**Dampak:**
- Backend API tidak bisa start
- Word Add-in mencoba fetch ke API yang tidak jalan
- Muncul error "Script error" di Word Add-in console

---

### 2. **Branch me2 Menggunakan Kode Lama**
Branch ini tidak memiliki improvements dari audit yang sudah saya lakukan:
- ❌ Tidak ada environment variable management
- ❌ Tidak ada logging framework (masih pakai print)
- ❌ File commands.js/html yang tidak berguna masih ada
- ❌ Duplicate code masih ada (tidak ada utils.py)

---

### 3. **Perubahan yang Dilakukan di Branch me2**

**File yang diubah:**
1. `word-addin/src/taskpane/taskpane.js`
   - Menambah deduplication logic (bagus!)
   - Menambah batching untuk handle dokumen besar (bagus!)
   - Menambah remove word functionality (bagus!)
   - API_URL di-hardcode kembali (tidak masalah, tapi tidak consistent)

2. `src/api/predictor.py`
   - Menambah `_expand_to_word_boundary()` - fix kata terpotong (bagus!)
   - Menambah `_clean_word()` - hapus tanda baca (bagus!)
   - Menambah `_is_numeric_only()` - skip angka (bagus!)
   - Kembali pakai `print()` bukan `logging` (tidak best practice)

3. `word-addin/src/taskpane/taskpane.css`
   - Kemungkinan styling updates

4. `data/italic_dataset.json`
   - Dataset updates

---

## ✅ SOLUSI LENGKAP

### **Step 1: Install Python Dependencies**

```bash
cd /home/user/Italic-Automation-System

# Install semua dependencies (WAJIB)
pip install -r requirements.txt

# Ini akan install:
# - fastapi, uvicorn (API server)
# - torch, transformers (ML model)
# - datasets, scikit-learn (data processing)
# - python-dotenv (environment variables)
# - dan library lainnya
```

**⏱️ Estimasi waktu:** 5-10 menit (tergantung koneksi internet)

---

### **Step 2: Cek Instalasi Berhasil**

```bash
# Test import libraries penting
python -c "import fastapi, torch, transformers; print('✅ All dependencies OK')"

# Jika berhasil, akan print: ✅ All dependencies OK
# Jika error, ulangi Step 1
```

---

### **Step 3: Start Backend API**

**Option A: Run di foreground (bisa lihat log)**
```bash
cd /home/user/Italic-Automation-System
python src/api/main.py
```

**Option B: Run di background**
```bash
cd /home/user/Italic-Automation-System
nohup python src/api/main.py > api.log 2>&1 &

# Cek log:
tail -f api.log
```

**Verifikasi API jalan:**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Jika berhasil, akan return JSON dengan status "healthy"
```

---

### **Step 4: Rebuild Word Add-in**

```bash
cd /home/user/Italic-Automation-System/word-addin

# Clear cache
rm -rf dist node_modules/.cache

# Rebuild
npm run build:dev
```

---

### **Step 5: Start Word Add-in Dev Server**

```bash
cd /home/user/Italic-Automation-System/word-addin
npm run dev-server
```

**Verifikasi dev server jalan:**
- Dev server akan jalan di `https://localhost:3000`
- Buka browser dan akses `https://localhost:3000/taskpane.html`
- Jika muncul page (walaupun error certificate), berarti server jalan

---

### **Step 6: Restart Word dan Load Add-in**

1. **Close Microsoft Word** sepenuhnya
2. **Delete Office Cache:**
   ```bash
   # Windows:
   # Hapus folder: %LOCALAPPDATA%\Microsoft\Office\16.0\Wef\

   # macOS:
   rm -rf ~/Library/Containers/com.microsoft.Word/Data/Library/Caches/

   # Linux: (jika ada)
   rm -rf ~/.cache/microsoft-word/
   ```
3. **Restart Word**
4. **Load Add-in:**
   ```bash
   cd /home/user/Italic-Automation-System/word-addin
   npm run start
   ```

---

## 🐛 TROUBLESHOOTING

### **Error: "Script error" di Word Add-in**

**Kemungkinan penyebab:**
1. ❌ Backend API tidak jalan → Cek dengan `curl http://localhost:8000/health`
2. ❌ CORS issue → Cek log API apakah ada CORS error
3. ❌ Word cache belum di-clear → Hapus cache dan restart Word
4. ❌ Dev server tidak jalan → Cek apakah `npm run dev-server` masih running

**Solusi:**
```bash
# 1. Pastikan API jalan
ps aux | grep "python.*main.py"

# 2. Pastikan dev server jalan
ps aux | grep "webpack.*dev-server"

# 3. Cek error di API log
tail -f api.log  # jika run di background

# 4. Cek error di browser console
# Buka dev tools di Word (F12 atau Ctrl+Shift+I)
```

---

### **Error: Model tidak ditemukan**

```
FileNotFoundError: models/indobert-italic/final/...
```

**Solusi:**
Model ML belum ada. Anda perlu:
1. Train model dulu: `python src/train.py`
2. Atau copy model dari backup jika ada

---

### **Error: PyTorch installation gagal**

**Solusi untuk Linux/macOS:**
```bash
# Install PyTorch CPU version (lebih cepat)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Solusi untuk Windows:**
```bash
# Install PyTorch dengan CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📋 CHECKLIST SEBELUM TESTING

- [ ] Python dependencies terinstall (`pip list | grep -E "fastapi|torch|transformers"`)
- [ ] Backend API jalan di port 8000 (`curl http://localhost:8000/health`)
- [ ] Model ML ada di `models/indobert-italic/final/`
- [ ] Word Add-in sudah di-build (`ls -la word-addin/dist/`)
- [ ] Dev server jalan di port 3000 (`curl -k https://localhost:3000/taskpane.html`)
- [ ] Word cache sudah di-clear
- [ ] Word sudah di-restart

---

## 🎯 TESTING CEPAT

Setelah semua setup:

1. Buka Word
2. Load add-in
3. Ketik test text di Word:
   ```
   Ini adalah contoh kalimat dengan kata back-end dan front-end yang harus italic.
   ```
4. Klik "Detect Italic" button
5. Cek console untuk error (F12)
6. Seharusnya mendeteksi "back-end" dan "front-end"

---

## 📞 JIKA MASIH ERROR

Share informasi berikut:

1. **Output dari commands ini:**
   ```bash
   python --version
   pip list | grep -E "fastapi|torch|transformers"
   curl http://localhost:8000/health
   ps aux | grep -E "python|node"
   ```

2. **Error message lengkap dari:**
   - Word Add-in console (F12)
   - API log (`tail -50 api.log`)
   - npm run dev-server output

3. **Screenshot error di Word Add-in**

---

## 💡 REKOMENDASI

Untuk menghindari masalah serupa di future:

1. **Gunakan virtual environment untuk Python:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Merge perubahan dari branch audit saya:**
   ```bash
   git checkout me2
   git merge claude/audit-italic-automation-Hzz1w
   # Resolve conflicts jika ada
   ```

3. **Gunakan Docker untuk consistency:**
   - Backend API di Docker container
   - Tidak perlu worry about dependencies

---

**Good luck! 🚀**
