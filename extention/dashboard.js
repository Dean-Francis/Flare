const API_BASE = "http://127.0.0.1:8000";

// ── Navigation ─────────────────────────────────────────────────────────────

const navItems   = document.querySelectorAll(".nav-item");
const views      = document.querySelectorAll(".view");
const topbarTitle = document.getElementById("topbarTitle");
const topbarSub   = document.getElementById("topbarSub");

const viewMeta = {
  dashboard: { title: "Flagged Emails", sub: "Emails you have manually flagged for training" },
  settings:  { title: "Settings",       sub: "Configure your Flare extension"                },
};

navItems.forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.view;

    navItems.forEach((b) => b.classList.remove("active"));
    views.forEach((v) => v.classList.remove("active"));

    btn.classList.add("active");
    document.getElementById(`view-${target}`).classList.add("active");

    topbarTitle.textContent = viewMeta[target].title;
    topbarSub.textContent   = viewMeta[target].sub;
  });
});

// ── Dashboard — flagged emails ─────────────────────────────────────────────

const emptyState    = document.getElementById("emptyState");
const emailTable    = document.getElementById("emailTable");
const emailTableBody = document.getElementById("emailTableBody");

// In-memory overrides: id → label int (visual only, resets on reload)
const localOverrides = {};

// Cached data + sort state
let cachedEmails = [];
let sortKey = "timestamp";
let sortDir = "desc";

function labelText(label) {
  return label === 1 ? "phishing" : "legitimate";
}

function formatTimestamp(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short", day: "numeric", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function formatConfidence(c) {
  if (c === null || c === undefined) return { text: "—", muted: true };
  return { text: `${(c * 100).toFixed(1)}%`, muted: false };
}

function sortEmails(emails) {
  const arr = [...emails];
  arr.sort((a, b) => {
    let av, bv;
    if (sortKey === "confidence") {
      av = a.confidence;
      bv = b.confidence;
      // Null values sink to the bottom regardless of direction
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
    } else {
      av = a.timestamp || "";
      bv = b.timestamp || "";
    }
    if (av < bv) return sortDir === "asc" ? -1 : 1;
    if (av > bv) return sortDir === "asc" ? 1 : -1;
    return 0;
  });
  return arr;
}

function updateSortArrows() {
  document.querySelectorAll("thead th.sortable").forEach((th) => {
    const arrow = th.querySelector(".sort-arrow");
    if (th.dataset.sort === sortKey) {
      th.classList.add("active");
      arrow.textContent = sortDir === "asc" ? "▲" : "▼";
    } else {
      th.classList.remove("active");
      arrow.textContent = "";
    }
  });
}

function renderTable() {
  emailTableBody.innerHTML = "";

  if (!cachedEmails || cachedEmails.length === 0) {
    emptyState.textContent = "No flagged emails yet.";
    emptyState.style.display = "";
    emailTable.style.display = "none";
    return;
  }

  emptyState.style.display = "none";
  emailTable.style.display = "";

  const sorted = sortEmails(cachedEmails);

  for (const email of sorted) {
    const effectiveLabel = localOverrides[email.id] ?? email.label;
    const labelStr = labelText(effectiveLabel);
    const conf = formatConfidence(email.confidence);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="cell-body" title="${escapeHtml(email.body)}">${escapeHtml(email.body)}</td>
      <td class="cell-confidence${conf.muted ? " muted" : ""}">${conf.text}</td>
      <td><span class="badge ${labelStr}">${labelStr}</span></td>
      <td class="cell-time">${formatTimestamp(email.timestamp)}</td>
      <td>
        <select class="override-select${localOverrides[email.id] !== undefined ? " overridden" : ""}" data-id="${email.id}">
          <option value="0"${effectiveLabel === 0 ? " selected" : ""}>Legitimate</option>
          <option value="1"${effectiveLabel === 1 ? " selected" : ""}>Phishing</option>
        </select>
      </td>
    `;

    const select = tr.querySelector("select");
    select.addEventListener("change", () => {
      const newLabel = parseInt(select.value, 10);
      localOverrides[email.id] = newLabel;
      select.classList.add("overridden");

      const badge = tr.querySelector(".badge");
      const newLabelStr = labelText(newLabel);
      badge.className = `badge ${newLabelStr}`;
      badge.textContent = newLabelStr;
    });

    emailTableBody.appendChild(tr);
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Sortable headers
document.querySelectorAll("thead th.sortable").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.sort;
    if (sortKey === key) {
      sortDir = sortDir === "asc" ? "desc" : "asc";
    } else {
      sortKey = key;
      sortDir = "desc";
    }
    updateSortArrows();
    renderTable();
  });
});

function loadFlaggedEmails(userId) {
  emptyState.textContent = "Loading…";
  emptyState.style.display = "";
  emailTable.style.display = "none";

  fetch(`${API_BASE}/flagged?user_id=${encodeURIComponent(userId)}`)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((data) => {
      cachedEmails = data.emails || [];
      updateSortArrows();
      renderTable();
    })
    .catch(() => {
      emptyState.textContent = "Could not reach the local server.";
      emptyState.style.display = "";
      emailTable.style.display = "none";
    });
}

// ── Settings ───────────────────────────────────────────────────────────────

const userIdInput   = document.getElementById("userIdInput");
const saveUserIdBtn = document.getElementById("saveUserIdBtn");
const saveFeedback  = document.getElementById("saveFeedback");

chrome.storage.local.get("userId", ({ userId }) => {
  if (userId) userIdInput.value = userId;
});

saveUserIdBtn.addEventListener("click", () => {
  const val = userIdInput.value.trim();
  chrome.storage.local.set({ userId: val || "anonymous" }, () => {
    saveFeedback.textContent = "Saved.";
    setTimeout(() => { saveFeedback.textContent = ""; }, 2000);
  });
});

// ── Init ───────────────────────────────────────────────────────────────────

chrome.storage.local.get("userId", ({ userId }) => {
  loadFlaggedEmails(userId || "anonymous");
});
