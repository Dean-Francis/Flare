const circle       = document.getElementById("statusCircle");
const label        = document.getElementById("statusLabel");
const iconCheck    = document.getElementById("iconCheck");
const iconX        = document.getElementById("iconX");
const flagBtn      = document.getElementById("flagBtn");
const manualToggle = document.getElementById("manualToggle");
const manualBody   = document.getElementById("manualBody");
const chevron      = document.getElementById("chevron");
const assessBtn    = document.getElementById("assessBtn");
const emailInput   = document.getElementById("emailInput");

// Current scan context — needed for flagging
let currentResult = null; // { predicted, confidence, body, cacheKey }

// ── Status circle helpers ──────────────────────────────────────────────────

function setIdle() {
  circle.className = "circle idle";
  iconCheck.style.display = "none";
  iconX.style.display     = "none";
  label.className  = "status-label";
  label.textContent = "No scan yet";
  flagBtn.classList.remove("visible");
  currentResult = null;
}

function setResult(result) {
  currentResult = result;
  const isManual = result.confidence === 1.0 && result.cacheKey;
  const detail = isManual ? "Manual" : `${Math.round(result.confidence * 100)}% confidence`;

  if (result.predicted === "legitimate") {
    circle.className = "circle safe";
    iconCheck.style.display = "";
    iconX.style.display     = "none";
    label.className  = "status-label safe";
    label.textContent = `Legitimate · ${detail}`;
    flagBtn.textContent = "Mark as Phishing";
  } else {
    circle.className = "circle phishing";
    iconCheck.style.display = "none";
    iconX.style.display     = "";
    label.className  = "status-label phishing";
    label.textContent = `Phishing · ${detail}`;
    flagBtn.textContent = "Mark as Legitimate";
  }

  flagBtn.classList.add("visible");
  flagBtn.disabled = false;
}

// ── Restore last result from storage on open ───────────────────────────────

chrome.storage.local.get("lastResult", ({ lastResult }) => {
  if (lastResult) {
    setResult(lastResult);
  } else {
    setIdle();
  }
});

// ── Dashboard button ───────────────────────────────────────────────────────

document.getElementById("dashboardBtn").addEventListener("click", () => {
  chrome.tabs.create({ url: chrome.runtime.getURL("dashboard.html") });
});

// ── Flag button ────────────────────────────────────────────────────────────

flagBtn.addEventListener("click", () => {
  if (!currentResult || !currentResult.body) return;

  const correctedLabel = currentResult.predicted === "phishing" ? 0 : 1;
  flagBtn.disabled = true;
  flagBtn.textContent = "Flagging…";

  chrome.runtime.sendMessage({
    type: "FLAG_EMAIL",
    body: currentResult.body,
    label: correctedLabel,
    cacheKey: currentResult.cacheKey,
  }, (response) => {
    if (chrome.runtime.lastError || !response?.ok) {
      flagBtn.disabled = false;
      flagBtn.textContent = correctedLabel === 1 ? "Mark as Phishing" : "Mark as Legitimate";
      label.textContent += " — flag failed";
      return;
    }

    // Flip the display to the corrected result
    const corrected = {
      predicted: correctedLabel === 1 ? "phishing" : "legitimate",
      confidence: 1.0,
      body: currentResult.body,
      cacheKey: currentResult.cacheKey,
    };
    setResult(corrected);
    flagBtn.disabled = true;
    flagBtn.textContent = "Flagged";
  });
});

// ── Manual check collapsible ───────────────────────────────────────────────

manualToggle.addEventListener("click", () => {
  const isOpen = manualBody.classList.toggle("open");
  chevron.classList.toggle("open", isOpen);
});

// ── Assess button ──────────────────────────────────────────────────────────

assessBtn.addEventListener("click", () => {
  const body = emailInput.value.trim();
  if (!body) return;

  assessBtn.disabled = true;
  assessBtn.textContent = "Checking…";

  chrome.runtime.sendMessage({ type: "MANUAL_CHECK", body }, (response) => {
    assessBtn.disabled = false;
    assessBtn.textContent = "Assess";

    if (chrome.runtime.lastError || !response) {
      label.className = "status-label";
      label.textContent = "Error — server unreachable";
      return;
    }

    setResult(response);
  });
});

// ── Live updates from background (auto scanned emails) ────────────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "SCAN_RESULT") {
    setResult(msg);
  }
});
