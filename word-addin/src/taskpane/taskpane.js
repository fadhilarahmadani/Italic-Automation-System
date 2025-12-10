/* global document, Office, Word, console */

// ============================================
// GLOBAL VARIABLES
// ============================================
let API_URL = "http://localhost:8000";
let detectedWords = [];
let stats = {
  detected: 0,
  formatted: 0,
  processingTime: 0,
};

// ============================================
// OFFICE.JS INITIALIZATION
// ============================================
Office.onReady((info) => {
  if (info.host === Office.HostType.Word) {
    console.log("✅ Office.js initialized");
    console.log("Host:", info.host);
    console.log("Platform:", info.platform);

    // Setup event listeners setelah DOM ready
    setupEventListeners();

    // Update initial status
    updateStatus("💡 Add-in siap digunakan. Mulai dengan mengetik teks di Word!", "info");

    console.log("✅ Italic Automation Add-in loaded successfully");
  }
});

// ============================================
// EVENT LISTENERS SETUP
// ============================================
function setupEventListeners() {
  // Button event listeners
  document.getElementById("detectBtn").onclick = detectItalicWords;
  document.getElementById("formatBtn").onclick = formatDocument;
  document.getElementById("clearBtn").onclick = clearAllItalics;

  // Threshold slider
  const thresholdSlider = document.getElementById("threshold");
  const thresholdValue = document.getElementById("thresholdValue");

  thresholdSlider.addEventListener("input", function () {
    const value = parseFloat(this.value).toFixed(2);
    thresholdValue.textContent = value;
  });

  // API URL input
  const apiUrlInput = document.getElementById("apiUrl");
  apiUrlInput.addEventListener("change", function () {
    API_URL = this.value.trim();
    console.log("API URL updated to:", API_URL);
    updateStatus("✅ API URL diperbarui: " + API_URL, "info");
  });

  console.log("✅ Event listeners setup complete");
}

// ============================================
// MAIN FUNCTIONS
// ============================================

/**
 * FUNCTION 1: Deteksi kata asing dalam dokumen
 */
async function detectItalicWords() {
  console.log("=".repeat(50));
  console.log("🔍 Starting detection...");

  updateStatus("🔍 Mendeteksi kata asing dalam dokumen...", "loading");
  setButtonsDisabled(true);

  try {
    await Word.run(async (context) => {
      // Step 1: Ambil teks dari dokumen Word
      console.log("📄 Getting document text...");
      const body = context.document.body;
      body.load("text");
      await context.sync();

      const documentText = body.text;
      console.log("Document text length:", documentText.length);
      console.log("First 100 chars:", documentText.substring(0, 100));

      // Validasi: cek dokumen tidak kosong
      if (!documentText || documentText.trim().length === 0) {
        throw new Error("Dokumen kosong. Silakan ketik teks terlebih dahulu.");
      }

      // Step 2: Ambil threshold setting
      const threshold = parseFloat(document.getElementById("threshold").value);
      console.log("Using threshold:", threshold);

      // Step 3: Kirim request ke API
      console.log("📡 Calling API:", API_URL + "/api/detect");

      const requestBody = {
        text: documentText,
        confidence_threshold: threshold,
      };

      const response = await fetch(API_URL + "/api/detect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      console.log("API Response status:", response.status);

      // Validasi response
      if (!response.ok) {
        const errorText = await response.text();
        console.error("API Error:", errorText);
        throw new Error(`API error (${response.status}): ${errorText}`);
      }

      // Step 4: Parse response
      const data = await response.json();
      console.log("API Response data:", data);

      if (!data.success) {
        throw new Error("API returned success=false");
      }

      // Step 5: Simpan hasil deteksi
      detectedWords = data.italic_words || [];
      stats.detected = data.total_detected || 0;
      stats.processingTime = data.processing_time || 0;

      console.log("✅ Detection complete:");
      console.log("  - Words detected:", stats.detected);
      console.log("  - Processing time:", stats.processingTime.toFixed(3), "seconds");

      // Step 6: Tampilkan hasil
      displayResults(detectedWords);
      updateStatistics();

      // Step 7: Update status
      if (stats.detected > 0) {
        updateStatus(`✅ Ditemukan ${stats.detected} kata asing yang perlu di-italic`, "success");
      } else {
        updateStatus("✅ Tidak ada kata asing yang terdeteksi dalam dokumen", "info");
      }
    });
  } catch (error) {
    console.error("❌ Error during detection:", error);

    let errorMessage = "Error: " + error.message;

    // Specific error messages
    if (error.message.includes("Failed to fetch")) {
      errorMessage = "❌ Tidak dapat terhubung ke API. Pastikan backend berjalan di " + API_URL;
    } else if (error.message.includes("NetworkError")) {
      errorMessage = "❌ Error jaringan. Cek koneksi internet dan API URL.";
    }

    updateStatus(errorMessage, "error");
  } finally {
    setButtonsDisabled(false);
    console.log("=".repeat(50));
  }
}

/**
 * FUNCTION 2: Format dokumen dengan italic otomatis
 */
async function formatDocument() {
  console.log("=".repeat(50));
  console.log("✨ Starting auto-format...");

  // Validasi: cek ada hasil deteksi
  if (detectedWords.length === 0) {
    updateStatus(
      "⚠️ Tidak ada kata yang terdeteksi. Jalankan 'Deteksi Kata Asing' terlebih dahulu.",
      "error"
    );
    return;
  }

  updateStatus("✨ Memformat dokumen dengan italic...", "loading");
  setButtonsDisabled(true);

  try {
    await Word.run(async (context) => {
      const body = context.document.body;
      let formattedCount = 0;

      console.log(`Processing ${detectedWords.length} words...`);

      // Loop setiap kata yang terdeteksi
      for (let i = 0; i < detectedWords.length; i++) {
        const wordInfo = detectedWords[i];
        console.log(`${i + 1}. Searching for: "${wordInfo.word}"`);

        // Search kata dalam dokumen
        const searchResults = body.search(wordInfo.word, {
          matchCase: false,
          matchWholeWord: false, // Biarkan false untuk tangkap phrase
        });

        searchResults.load("font, text");
        await context.sync();

        console.log(`   Found ${searchResults.items.length} occurrence(s)`);

        // Apply italic ke semua occurrence
        for (let j = 0; j < searchResults.items.length; j++) {
          searchResults.items[j].font.italic = true;
          formattedCount++;
        }
      }

      await context.sync();

      console.log(`✅ Formatted ${formattedCount} word occurrence(s)`);

      // Update statistics
      stats.formatted = formattedCount;
      updateStatistics();

      updateStatus(`✅ Berhasil memformat ${formattedCount} kata dengan italic`, "success");
    });
  } catch (error) {
    console.error("❌ Error during formatting:", error);
    updateStatus("❌ Error saat memformat: " + error.message, "error");
  } finally {
    setButtonsDisabled(false);
    console.log("=".repeat(50));
  }
}

/**
 * FUNCTION 3: Hapus semua italic formatting
 */
async function clearAllItalics() {
  console.log("=".repeat(50));
  console.log("🗑️ Clearing all italics...");

  // Konfirmasi user
  const confirmClear = confirm(
    "Apakah Anda yakin ingin menghapus semua format italic dari dokumen?"
  );

  if (!confirmClear) {
    console.log("User cancelled clear operation");
    return;
  }

  updateStatus("🗑️ Menghapus semua format italic...", "loading");
  setButtonsDisabled(true);

  try {
    await Word.run(async (context) => {
      const body = context.document.body;

      // Remove italic dari seluruh dokumen
      body.font.italic = false;

      await context.sync();

      console.log("✅ All italics cleared");

      // Reset statistics
      stats.formatted = 0;
      updateStatistics();

      updateStatus("✅ Semua format italic telah dihapus dari dokumen", "success");
    });
  } catch (error) {
    console.error("❌ Error during clear:", error);
    updateStatus("❌ Error saat menghapus italic: " + error.message, "error");
  } finally {
    setButtonsDisabled(false);
    console.log("=".repeat(50));
  }
}

// ============================================
// UI UPDATE FUNCTIONS
// ============================================

/**
 * Tampilkan hasil deteksi di UI
 */
function displayResults(words) {
  console.log("Displaying results for", words.length, "words");

  const resultsSection = document.getElementById("resultsSection");
  const resultsList = document.getElementById("resultsList");

  // Hide section jika tidak ada hasil
  if (words.length === 0) {
    resultsSection.style.display = "none";
    return;
  }

  // Show section
  resultsSection.style.display = "block";

  // Clear previous results
  resultsList.innerHTML = "";

  // Create result items
  words.forEach((wordInfo, index) => {
    const item = document.createElement("div");
    item.className = "result-item";

    // Word text
    const wordDiv = document.createElement("div");
    wordDiv.className = "result-word";
    wordDiv.textContent = wordInfo.word;

    // Confidence
    const confidenceDiv = document.createElement("div");
    confidenceDiv.className = "result-confidence";

    const confidencePercent = (wordInfo.confidence * 100).toFixed(1);

    confidenceDiv.innerHTML = `
            <span class="confidence-badge">${confidencePercent}%</span>
            <span>confidence • Label: ${wordInfo.label}</span>
        `;

    item.appendChild(wordDiv);
    item.appendChild(confidenceDiv);
    resultsList.appendChild(item);
  });

  console.log("✅ Results displayed in UI");
}

/**
 * Update statistik di UI
 */
function updateStatistics() {
  document.getElementById("detectedCount").textContent = stats.detected;
  document.getElementById("formattedCount").textContent = stats.formatted;

  const timeText = stats.processingTime > 0 ? stats.processingTime.toFixed(2) + "s" : "-";
  document.getElementById("processingTime").textContent = timeText;

  console.log("Statistics updated:", stats);
}

/**
 * Update status box
 */
function updateStatus(message, type = "info") {
  const statusBox = document.getElementById("statusBox");
  statusBox.textContent = message;

  // Remove all status classes
  statusBox.className = "status-box";

  // Add new status class
  statusBox.classList.add("status-" + type);

  console.log(`Status: [${type}] ${message}`);
}

/**
 * Enable/disable semua tombol
 */
function setButtonsDisabled(disabled) {
  document.getElementById("detectBtn").disabled = disabled;
  document.getElementById("formatBtn").disabled = disabled;
  document.getElementById("clearBtn").disabled = disabled;

  // Tambahkan spinner ke tombol yang aktif saat loading
  if (disabled) {
    const detectBtn = document.getElementById("detectBtn");
    if (!detectBtn.querySelector(".spinner")) {
      const spinner = document.createElement("span");
      spinner.className = "spinner";
      detectBtn.insertBefore(spinner, detectBtn.firstChild);
    }
  } else {
    // Remove spinner
    const spinners = document.querySelectorAll(".spinner");
    spinners.forEach((s) => s.remove());
  }
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Test API connection
 */
async function testAPIConnection() {
  try {
    console.log("Testing API connection to:", API_URL);
    const response = await fetch(API_URL + "/health");
    const data = await response.json();
    console.log("API health check:", data);
    return data.status === "healthy";
  } catch (error) {
    console.error("API connection test failed:", error);
    return false;
  }
}

// ============================================
// INITIALIZATION LOG
// ============================================
console.log("=".repeat(50));
console.log("🔤 Italic Automation Add-in");
console.log("Version: 1.0.0");
console.log("API URL:", API_URL);
console.log("=".repeat(50));
