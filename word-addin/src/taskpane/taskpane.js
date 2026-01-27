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
   HELPER: Detect Reference Paragraph (Indonesia format)
===================== */
function isReferenceParagraph(text, style) {
  // Check style first
  if (style === "Bibliography" || /referensi|pustaka/i.test(style)) {
    return true;
  }

  // Pattern detection untuk format Indonesia
  const patterns = [
    /^\[\d+\]/,                    // [1], [2]
    /^\d+\.\s+[A-Z]/,              // 1. Smith
    /^[A-Z][a-z]+,\s+[A-Z]\./,     // Koto, F.
    /,\s*dkk\.?\s*\d{4}/i,         // , dkk. 2020
    /dkk\.?\s*\(\d{4}\)/i,         // dkk. (2020)
    /\.\s*\d{4}\./,                // . 2020.
    /\(\d{4}\)\./,                 // (2020).
    /https?:\/\//,                 // URLs
    /DOI:/i,
    /doi\.org/i,
    /arXiv:/i,
    /Proceedings of/i,
    /hlm\.\s+\d+/i,                // hlm. 1-10
    /Vol\.\s+\d+/i,
    /Penerbit/i,
    /Jakarta|Bandung|Yogyakarta|Surabaya|Semarang/i
  ];

  return patterns.some(pattern => pattern.test(text));
}

/* =====================
   DETECTION (WITH SMART FILTERING)
===================== */
async function detectItalic() {
  try {
    const startTime = performance.now();
    detectedSpans = [];
    uniqueWords = [];

    // Read filter preferences
    const skipHeaders = document.getElementById("skipHeaders").checked;
    const skipReferences = document.getElementById("skipReferences").checked;

    updateStatus("🔍 Mendeteksi kata asing...", "info");

    await Word.run(async (context) => {
      const paragraphs = context.document.body.paragraphs;
      paragraphs.load("items");
      await context.sync();

      // Load styles and text for filtering
      paragraphs.items.forEach(p => {
        p.load("style");
        p.load("text");
      });
      await context.sync();

      // STEP 1: Detect "DAFTAR PUSTAKA" section
      let referenceStartIndex = -1;

      if (skipReferences) {
        for (let i = 0; i < paragraphs.items.length; i++) {
          const para = paragraphs.items[i];
          const text = para.text.trim();
          const style = para.style;

          // Detect "DAFTAR PUSTAKA" heading
          if ((style === "Heading 1" || style === "Heading 2" || style === "Heading 3") &&
              /^(DAFTAR\s+PUSTAKA|REFERENSI|BIBLIOGRAPHY)/i.test(text)) {
            referenceStartIndex = i;
            console.log(`📚 Found "DAFTAR PUSTAKA" at paragraph ${i}: "${text}"`);
            break;
          }
        }
      }

      // STEP 2: Filter paragraphs
      const paragraphsToProcess = [];
      const skippedCount = { headers: 0, references: 0 };

      for (let i = 0; i < paragraphs.items.length; i++) {
        const para = paragraphs.items[i];
        const style = para.style;
        const text = para.text.trim();
        let shouldSkip = false;

        // Skip empty paragraphs
        if (text.length === 0) {
          continue;
        }

        // FILTER 1: Skip Headers
        if (skipHeaders &&
            (style === "Heading 1" ||
             style === "Heading 2" ||
             style === "Heading 3" ||
             style === "Title" ||
             style === "Subtitle")) {
          shouldSkip = true;
          skippedCount.headers++;
          console.log(`🚫 Skipped header [${style}]: "${text.substring(0, 50)}..."`);
        }

        // FILTER 2: Skip References (Indonesia format)
        if (skipReferences && !shouldSkip) {
          // Method A: Skip everything after "DAFTAR PUSTAKA" heading
          if (referenceStartIndex !== -1 && i >= referenceStartIndex) {
            shouldSkip = true;
            skippedCount.references++;
            if (i === referenceStartIndex) {
              console.log(`📚 Skipping "DAFTAR PUSTAKA" section starting at paragraph ${i}`);
            }
          }

          // Method B: Pattern-based detection (backup)
          if (!shouldSkip && isReferenceParagraph(text, style)) {
            shouldSkip = true;
            skippedCount.references++;
            console.log(`📄 Skipped reference entry: "${text.substring(0, 60)}..."`);
          }
        }

        // Add to processing list if not skipped
        if (!shouldSkip) {
          paragraphsToProcess.push({
            index: i,
            text: text,
            paragraph: para
          });
        }
      }

      console.log(`📊 Filter Summary:`);
      console.log(`  Total paragraphs: ${paragraphs.items.length}`);
      console.log(`  To process: ${paragraphsToProcess.length}`);
      console.log(`  Skipped headers: ${skippedCount.headers}`);
      console.log(`  Skipped references: ${skippedCount.references}`);

      // Update status with filter info
      if (skippedCount.headers > 0 || skippedCount.references > 0) {
        updateStatus(
          `🔍 Memproses ${paragraphsToProcess.length}/${paragraphs.items.length} paragraf ` +
          `(lewati: ${skippedCount.headers} header, ${skippedCount.references} referensi)`,
          "info"
        );
      }

      // STEP 3: Process filtered paragraphs
      const texts = paragraphsToProcess.map(p => p.text);
      const threshold = parseFloat(document.getElementById("threshold").value);

      // Batching: API maksimal 100 paragraphs per request
      const BATCH_SIZE = 100;
      const totalBatches = Math.ceil(texts.length / BATCH_SIZE);
      
      console.log(`Total paragraphs: ${texts.length}, batches: ${totalBatches}`);

      // Process batches
      for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
        const start = batchIndex * BATCH_SIZE;
        const end = Math.min(start + BATCH_SIZE, texts.length);
        const batchTexts = texts.slice(start, end);

        updateStatus(
          `🔍 Mendeteksi kata asing... (batch ${batchIndex + 1}/${totalBatches})`,
          "info"
        );

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

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

      // Adjust paragraph index untuk batch AND filtered paragraphs
      data.results.forEach((para) => {
        if (para.italic_words && Array.isArray(para.italic_words)) {
          // Get original paragraph index from paragraphsToProcess
          const originalIndex = paragraphsToProcess[start + para.paragraph_index].index;

          para.italic_words.forEach((w) => {
            detectedSpans.push({
              paragraphIndex: originalIndex, // Use original paragraph index!
              start: w.start_pos,
              end: w.end_pos,
              word: w.word,
              confidence: w.confidence,
            });
          });
        }
      });
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
