/* global document, Office, Word */

let API_URL = "http://localhost:8000";
let detectedSpans = [];
let formattedCount = 0;

Office.onReady((info) => {
  if (info.host === Office.HostType.Word) {
    const detectBtn = document.getElementById("detectBtn");
    const formatBtn = document.getElementById("formatBtn");
    const clearBtn = document.getElementById("clearBtn");

    if (detectBtn) detectBtn.onclick = detectItalic;
    if (formatBtn) formatBtn.onclick = applyItalic;
    if (clearBtn) clearBtn.onclick = clearItalic;

    // Threshold slider update
    const thresholdSlider = document.getElementById("threshold");
    const thresholdValue = document.getElementById("thresholdValue");
    if (thresholdSlider && thresholdValue) {
      thresholdSlider.oninput = function () {
        thresholdValue.textContent = parseFloat(this.value).toFixed(2);
      };
    }

    // API URL update
    const apiUrlInput = document.getElementById("apiUrl");
    if (apiUrlInput) {
      apiUrlInput.onchange = function () {
        API_URL = this.value.trim();
        updateStatus("API URL diubah ke: " + API_URL, "info");
      };
    }
  }
});

// Update status box
function updateStatus(message, type = "info") {
  const statusBox = document.getElementById("statusBox");
  if (statusBox) {
    statusBox.textContent = message;
    statusBox.className = `status-box status-${type}`;
  }
}

// Update statistics
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

async function detectItalic() {
  const startTime = performance.now();
  detectedSpans = [];
  updateStatus("🔍 Mendeteksi kata asing...", "info");

  await Word.run(async (context) => {
    const paragraphs = context.document.body.paragraphs;
    paragraphs.load("items/text");
    await context.sync();

    const texts = paragraphs.items.map((p) => p.text);
    const threshold = parseFloat(document.getElementById("threshold").value);

    const res = await fetch(API_URL + "/api/batch-detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        paragraphs: texts,
        confidence_threshold: threshold,
      }),
    });

    const data = await res.json();

    data.results.forEach((para) => {
      para.italic_words.forEach((w) => {
        detectedSpans.push({
          paragraphIndex: para.paragraph_index,
          start: w.start_pos,
          end: w.end_pos,
          word: w.word,
          confidence: w.confidence,
        });
      });
    });

    const endTime = performance.now();
    const processingTime = ((endTime - startTime) / 1000).toFixed(2) + "s";

    updateStats(detectedSpans.length, formattedCount, processingTime);
    updateStatus(`✅ Ditemukan ${detectedSpans.length} kata asing`, "success");
    showDetectedResults(detectedSpans);
  }).catch((error) => {
    console.error("Error detecting:", error);
    updateStatus("❌ Gagal mendeteksi: " + error.message, "error");
  });
}

async function applyItalic() {
  const startTime = performance.now();
  formattedCount = 0; // Reset counter

  if (detectedSpans.length === 0) {
    updateStatus("⚠️ Tidak ada kata yang terdeteksi untuk di-italic", "warning");
    return;
  }

  updateStatus("✨ Menerapkan italic...", "info");

  await Word.run(async (context) => {
    const paragraphs = context.document.body.paragraphs;
    paragraphs.load("items");
    await context.sync();

    // Group by paragraph
    const spansByPara = {};
    detectedSpans.forEach((span) => {
      if (!spansByPara[span.paragraphIndex]) {
        spansByPara[span.paragraphIndex] = [];
      }
      spansByPara[span.paragraphIndex].push(span);
    });

    // Process tiap paragraph
    for (const paraIndexStr in spansByPara) {
      const paraIndex = parseInt(paraIndexStr);
      const spans = spansByPara[paraIndexStr];
      const para = paragraphs.items[paraIndex];

      para.load("text");
      await context.sync();

      const originalText = para.text;

      // Track posisi untuk handle duplicate words
      const wordPositions = {};

      spans.forEach((span) => {
        const word = originalText.substring(span.start, span.end);
        if (!wordPositions[word]) {
          wordPositions[word] = [];
        }
        wordPositions[word].push(span.start);
      });

      // Process setiap unique word
      for (const word in wordPositions) {
        const positions = wordPositions[word];

        // Search semua occurences
        const searchResults = para.search(word, {
          matchCase: true,
          matchWholeWord: false,
        });
        searchResults.load("items");
        await context.sync();

        // Untuk setiap search result, cek apakah posisinya match
        let foundCount = 0;
        let searchIndex = 0;

        // Hitung posisi setiap occurrence di original text
        const allOccurrences = [];
        let searchStart = 0;
        while (true) {
          const idx = originalText.indexOf(word, searchStart);
          if (idx === -1) break;
          allOccurrences.push(idx);
          searchStart = idx + 1;
        }

        // Apply italic hanya pada occurrence yang match dengan detected positions
        searchResults.items.forEach((result, idx) => {
          if (idx < allOccurrences.length) {
            const occurrencePos = allOccurrences[idx];
            if (positions.includes(occurrencePos)) {
              result.font.italic = true;
              formattedCount++;
            }
          }
        });
      }

      await context.sync();
    }

    const endTime = performance.now();
    const processingTime = ((endTime - startTime) / 1000).toFixed(2) + "s";

    updateStats(detectedSpans.length, formattedCount, processingTime);
    updateStatus(`✅ Berhasil menerapkan italic pada ${formattedCount} kata`, "success");
  }).catch((error) => {
    console.error("Error applying italic:", error);
    updateStatus("❌ Gagal menerapkan italic: " + error.message, "error");
  });
}

// Clear all italic formatting
async function clearItalic() {
  updateStatus("🗑️ Menghapus semua italic...", "info");

  await Word.run(async (context) => {
    const body = context.document.body;
    body.load("text");
    await context.sync();

    // Get all content and remove italic
    const paragraphs = body.paragraphs;
    paragraphs.load("items");
    await context.sync();

    paragraphs.items.forEach((para) => {
      para.font.italic = false;
    });

    await context.sync();

    // Reset counters
    detectedSpans = [];
    formattedCount = 0;
    updateStats(0, 0, "-");

    const section = document.getElementById("resultsSection");
    if (section) section.style.display = "none";

    updateStatus("✅ Semua italic telah dihapus", "success");
  }).catch((error) => {
    console.error("Error clearing italic:", error);
    updateStatus("❌ Gagal menghapus italic: " + error.message, "error");
  });
}

function showDetectedResults(spans) {
  const list = document.getElementById("resultsList");
  const section = document.getElementById("resultsSection");

  list.innerHTML = "";
  if (spans.length === 0) {
    section.style.display = "none";
    return;
  }

  spans.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = `${s.word} (${(s.confidence * 100).toFixed(1)}%)`;
    list.appendChild(li);
  });

  section.style.display = "block";
}
