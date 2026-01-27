/* global document, Office, Word */

let API_URL = "http://localhost:8000";
let detectedSpans = [];      // Semua span (untuk apply italic)
let uniqueWords = [];        // Kata unik (untuk tampilan UI)
let formattedCount = 0;

const HIGHLIGHT_COLOR = "#90EE90";

// Global error handler untuk mencegah uncaught errors
window.addEventListener('error', function(event) {
  console.error('Global Error:', event.error);
  updateStatus('❌ Error: ' + (event.error?.message || 'Terjadi kesalahan'), 'error');
  event.preventDefault();
});

// Global unhandled rejection handler
window.addEventListener('unhandledrejection', function(event) {
  console.error('Unhandled Promise Rejection:', event.reason);
  updateStatus('❌ Error: ' + (event.reason?.message || 'Koneksi gagal. Pastikan API berjalan.'), 'error');
  event.preventDefault();
});

Office.onReady((info) => {
  if (info.host === Office.HostType.Word) {
    // Check API connection on startup
    checkApiConnection();
    
    const detectBtn = document.getElementById("detectBtn");
    const formatBtn = document.getElementById("formatBtn");
    const clearBtn = document.getElementById("clearBtn");

    if (detectBtn) detectBtn.onclick = detectItalic;
    if (formatBtn) formatBtn.onclick = applyItalic;
    if (clearBtn) clearBtn.onclick = clearItalic;

    const thresholdSlider = document.getElementById("threshold");
    const thresholdValue = document.getElementById("thresholdValue");
    if (thresholdSlider && thresholdValue) {
      thresholdSlider.oninput = function () {
        thresholdValue.textContent = parseFloat(this.value).toFixed(2);
      };
    }
  }
});

/* =====================
   API CONNECTION CHECK
===================== */
async function checkApiConnection() {
  try {
    updateStatus("🔄 Memeriksa koneksi API...", "info");
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch(API_URL + "/health", {
      method: "GET",
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      updateStatus("✅ API terhubung dan siap", "success");
      return true;
    } else {
      updateStatus("⚠️ API tidak merespons dengan benar", "warning");
      return false;
    }
  } catch (error) {
    console.error("API connection check failed:", error);
    if (error.name === 'AbortError') {
      updateStatus("❌ API tidak dapat dijangkau (timeout). Pastikan server berjalan di " + API_URL, "error");
    } else {
      updateStatus("❌ Tidak dapat terhubung ke API. Pastikan server berjalan di " + API_URL, "error");
    }
    return false;
  }
}

/* =====================
   STATUS & STATISTICS
===================== */
function updateStatus(message, type = "info") {
  try {
    const statusBox = document.getElementById("statusBox");
    if (statusBox) {
      statusBox.textContent = message;
      statusBox.className = `status-box status-${type}`;
    }
  } catch (error) {
    console.error("Error updating status:", error);
  }
}

function updateStats(detected = null, formatted = null, time = null) {
  if (detected !== null) {
    const el = document.getElementById("detectedCount");
    if (el) el.textContent = detected;
  }
  if (formatted !== null) {
    const el = document.getElementById("formattedCount");
    if (el) el.textContent = formatted;
  }
  if (time !== null) {
    const el = document.getElementById("processingTime");
    if (el) el.textContent = time;
  }
}

/* =====================
   DETECTION
===================== */
async function detectItalic() {
  try {
    const startTime = performance.now();
    detectedSpans = [];
    uniqueWords = [];

    updateStatus("🔍 Mendeteksi kata asing...", "info");

    await Word.run(async (context) => {
      const paragraphs = context.document.body.paragraphs;
      paragraphs.load("items");
      await context.sync();

      // Load text for all paragraphs
      paragraphs.items.forEach(p => p.load("text"));
      await context.sync();

      // Collect non-empty paragraphs with their original indices
      const texts = [];
      const paragraphIndexMap = []; // Maps text array index to original paragraph index

      for (let i = 0; i < paragraphs.items.length; i++) {
        const text = paragraphs.items[i].text.trim();
        if (text.length > 0) {
          texts.push(text);
          paragraphIndexMap.push(i); // Store original paragraph index
        }
      }

      console.log(`📄 Processing ${texts.length} paragraphs`);
      const threshold = parseFloat(document.getElementById("threshold").value);

      // Dynamic batch size based on document size
      // Larger batches = fewer API calls, but may timeout
      // Smaller batches = more API calls, but more reliable
      let BATCH_SIZE = 100;
      if (texts.length > 500) {
        BATCH_SIZE = 50; // Smaller batches for very large documents
        console.log(`⚠️ Large document detected (${texts.length} paragraphs). Using smaller batch size: ${BATCH_SIZE}`);
      }

      const totalBatches = Math.ceil(texts.length / BATCH_SIZE);
      console.log(`📦 Processing: ${texts.length} paragraphs in ${totalBatches} batches`);

      // Warn user if document is large
      if (totalBatches > 3) {
        updateStatus(
          `⏳ Dokumen besar (${texts.length} paragraf, ${totalBatches} batch). Mohon tunggu 1-3 menit...`,
          "info"
        );
      }

      // Process batches
      for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
        const start = batchIndex * BATCH_SIZE;
        const end = Math.min(start + BATCH_SIZE, texts.length);
        const batchTexts = texts.slice(start, end);

        const batchStartTime = performance.now();
        const estimatedTime = totalBatches > 1 ? ` (~${totalBatches * 20}s total)` : '';

        updateStatus(
          `🔍 Batch ${batchIndex + 1}/${totalBatches} (${batchTexts.length} paragraf)${estimatedTime}`,
          "info"
        );

        const controller = new AbortController();
        // INCREASED TIMEOUT: 90 seconds per batch for large documents (8000+ words)
        const timeoutId = setTimeout(() => controller.abort(), 90000);

        try {
          const res = await fetch(API_URL + "/api/batch-detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              paragraphs: batchTexts,
              confidence_threshold: threshold,
            }),
            signal: controller.signal
          });

          clearTimeout(timeoutId);

          // Cek status HTTP
          if (!res.ok) {
            const errorText = await res.text();
            console.error("API Error Response:", errorText);
            throw new Error(`API error (${res.status}): ${errorText}`);
          }

          const batchTime = ((performance.now() - batchStartTime) / 1000).toFixed(1);
          console.log(`✅ Batch ${batchIndex + 1} completed in ${batchTime}s`);

          // Parse response JSON
          const data = await res.json();
          console.log(`Batch ${batchIndex + 1} Response:`, data);

          // Validasi struktur data dengan lebih detail
          if (!data || !data.success) {
            console.error("API Response:", data);
            throw new Error("API mengembalikan status tidak berhasil");
          }

          if (!data.results || !Array.isArray(data.results)) {
            console.error("Invalid data structure:", data);
            throw new Error("Format respons API tidak valid - results tidak ditemukan");
          }

          // Adjust paragraph index untuk batch
          data.results.forEach((para) => {
            if (para.italic_words && Array.isArray(para.italic_words)) {
              // Get original paragraph index from map
              const originalIndex = paragraphIndexMap[start + para.paragraph_index];

              para.italic_words.forEach((w) => {
                detectedSpans.push({
                  paragraphIndex: originalIndex,
                  start: w.start_pos,
                  end: w.end_pos,
                  word: w.word,
                  confidence: w.confidence,
                });
              });
            }
          });
        } catch (error) {
          clearTimeout(timeoutId);
          if (error.name === 'AbortError') {
            throw new Error(
              `Timeout pada batch ${batchIndex + 1}/${totalBatches}. ` +
              `Dokumen terlalu besar (${texts.length} paragraf). ` +
              `Coba bagi dokumen menjadi beberapa bagian.`
            );
          }
          throw error;
        }
      }

    // Deduplicate: Kelompokkan kata unik dengan confidence tertinggi dan hitung jumlah kemunculan
    const wordMap = new Map();
    detectedSpans.forEach((span) => {
      const wordLower = span.word.toLowerCase();
      if (!wordMap.has(wordLower)) {
        wordMap.set(wordLower, {
          word: span.word,
          confidence: span.confidence,
          count: 1,
        });
      } else {
        const existing = wordMap.get(wordLower);
        existing.count++;
        // Simpan confidence tertinggi
        if (span.confidence > existing.confidence) {
          existing.confidence = span.confidence;
          existing.word = span.word; // Simpan case asli dengan confidence tertinggi
        }
      }
    });

    // Convert Map ke array untuk UI
    uniqueWords = Array.from(wordMap.values());

    // Sort by confidence descending
    uniqueWords.sort((a, b) => b.confidence - a.confidence);

    const endTime = performance.now();
    const processingTime = ((endTime - startTime) / 1000).toFixed(2) + "s";

    updateStats(detectedSpans.length, formattedCount, processingTime);
    updateStatus(
      `✅ Ditemukan ${uniqueWords.length} kata asing unik (${detectedSpans.length} kemunculan) - KBBI filtered di backend`,
      "success"
    );
    showDetectedResults(uniqueWords);
    }).catch((error) => {
      console.error("Word.run error:", error);
      if (error.name === 'AbortError') {
        updateStatus("❌ Request timeout. Server mungkin terlalu lambat atau tidak merespons.", "error");
      } else {
        updateStatus("❌ Gagal mendeteksi: " + error.message, "error");
      }
    });
  } catch (error) {
    console.error("Detect error:", error);
    updateStatus("❌ Error: " + (error.message || "Terjadi kesalahan"), "error");
  }
}

/* =====================
   APPLY ITALIC + HIGHLIGHT
===================== */
async function applyItalic() {
  try {
    const startTime = performance.now();
    formattedCount = 0;

    if (detectedSpans.length === 0) {
      updateStatus("⚠️ Tidak ada kata yang terdeteksi", "warning");
      return;
    }

    updateStatus("✨ Menerapkan italic & highlight...", "info");

    await Word.run(async (context) => {
      const paragraphs = context.document.body.paragraphs;
      paragraphs.load("items");
      await context.sync();

      const spansByPara = {};
      detectedSpans.forEach((span) => {
        if (!spansByPara[span.paragraphIndex]) {
          spansByPara[span.paragraphIndex] = [];
        }
        spansByPara[span.paragraphIndex].push(span);
      });

      for (const paraIndexStr in spansByPara) {
        const paraIndex = parseInt(paraIndexStr);
        const spans = spansByPara[paraIndexStr];
        const para = paragraphs.items[paraIndex];

        para.load("text");
        await context.sync();

        const originalText = para.text;
        const wordPositions = {};

        spans.forEach((span) => {
          const word = originalText.substring(span.start, span.end);
          if (!wordPositions[word]) wordPositions[word] = [];
          wordPositions[word].push(span.start);
        });

        for (const word in wordPositions) {
          const positions = wordPositions[word];

          const searchResults = para.search(word, {
            matchCase: true,
            matchWholeWord: false,
          });
          searchResults.load("items");
          await context.sync();

          const allOccurrences = [];
          let searchStart = 0;
          while (true) {
            const idx = originalText.indexOf(word, searchStart);
            if (idx === -1) break;
            allOccurrences.push(idx);
            searchStart = idx + 1;
          }

          searchResults.items.forEach((result, idx) => {
            if (idx < allOccurrences.length) {
              const occurrencePos = allOccurrences[idx];
              if (positions.includes(occurrencePos)) {
                result.font.italic = true;
                result.font.highlightColor = HIGHLIGHT_COLOR; // ✅ HIGHLIGHT
                formattedCount++;
              }
            }
          });
        }
      }

      const endTime = performance.now();
      const processingTime = ((endTime - startTime) / 1000).toFixed(2) + "s";

      updateStats(detectedSpans.length, formattedCount, processingTime);
      updateStatus(
        `✅ Italic & highlight diterapkan pada ${formattedCount} kata`,
        "success"
      );
    }).catch((error) => {
      console.error("Word.run error:", error);
      updateStatus("❌ Gagal menerapkan format: " + error.message, "error");
    });
  } catch (error) {
    console.error("Apply italic error:", error);
    updateStatus("❌ Error: " + (error.message || "Terjadi kesalahan"), "error");
  }
}

/* =====================
   CLEAR HIGHLIGHT ONLY
===================== */
async function clearItalic() {
  try {
    updateStatus("🧹 Menghapus highlight...", "info");

    await Word.run(async (context) => {
      const paragraphs = context.document.body.paragraphs;
      paragraphs.load("items");
      await context.sync();

      paragraphs.items.forEach((para) => {
        para.font.highlightColor = null;
      });

      await context.sync();

      updateStatus("✅ Highlight berhasil dihapus", "success");
    }).catch((error) => {
      console.error("Word.run error:", error);
      updateStatus("❌ Gagal menghapus highlight: " + error.message, "error");
    });
  } catch (error) {
    console.error("Clear highlight error:", error);
    updateStatus("❌ Error: " + (error.message || "Terjadi kesalahan"), "error");
  }
}

/* =====================
   UI RESULTS (WITH DEDUPLICATION)
===================== */
function showDetectedResults(words) {
  const list = document.getElementById("resultsList");
  const section = document.getElementById("resultsSection");

  list.innerHTML = "";
  if (words.length === 0) {
    section.style.display = "none";
    return;
  }

  words.forEach((item, index) => {
    const li = document.createElement("li");
    li.className = "result-item";
    li.setAttribute("data-index", index);

    // Container untuk word dan tombol hapus
    const itemContent = document.createElement("div");
    itemContent.className = "result-item-content";

    // Word dan confidence
    const wordInfo = document.createElement("div");
    wordInfo.className = "result-word-info";

    const wordSpan = document.createElement("span");
    wordSpan.className = "result-word";
    // Tampilkan kata dengan jumlah kemunculan jika lebih dari 1
    wordSpan.textContent = item.count > 1 ? `${item.word} (×${item.count})` : item.word;

    const confidenceSpan = document.createElement("span");
    confidenceSpan.className = "confidence-badge";
    confidenceSpan.textContent = `${(item.confidence * 100).toFixed(1)}%`;

    wordInfo.appendChild(wordSpan);
    wordInfo.appendChild(confidenceSpan);

    // Tombol hapus (x)
    const removeBtn = document.createElement("button");
    removeBtn.className = "remove-btn";
    removeBtn.innerHTML = "✕";
    removeBtn.title = "Hapus kata ini dari daftar";
    removeBtn.onclick = function (e) {
      e.stopPropagation();
      removeDetectedWord(index);
    };

    itemContent.appendChild(wordInfo);
    itemContent.appendChild(removeBtn);
    li.appendChild(itemContent);
    list.appendChild(li);
  });

  section.style.display = "block";
}

/* =====================
   REMOVE DETECTED WORD (WITH DEDUPLICATION)
===================== */
function removeDetectedWord(index) {
  if (index >= 0 && index < uniqueWords.length) {
    const removedWord = uniqueWords[index].word;
    const removedWordLower = removedWord.toLowerCase();
    
    // Hapus dari uniqueWords
    uniqueWords.splice(index, 1);
    
    // Hapus SEMUA kemunculan kata ini dari detectedSpans
    detectedSpans = detectedSpans.filter(
      (span) => span.word.toLowerCase() !== removedWordLower
    );

    // Update UI
    updateStats(detectedSpans.length, formattedCount, null);
    showDetectedResults(uniqueWords);

    // Update status
    updateStatus(
      `🗑️ "${removedWord}" dihapus. Tersisa ${uniqueWords.length} kata unik (${detectedSpans.length} kemunculan).`,
      "info"
    );
  }
}
