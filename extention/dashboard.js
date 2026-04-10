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

function renderTable(emails) {
  emailTableBody.innerHTML = "";

  if (!emails || emails.length === 0) {
    emptyState.textContent = "No flagged emails yet.";
    emptyState.style.display = "";
    emailTable.style.display = "none";
    return;
  }

  emptyState.style.display = "none";
  emailTable.style.display = "";

  for (const email of emails) {
    const effectiveLabel = localOverrides[email.id] ?? email.label;
    const labelStr = labelText(effectiveLabel);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="cell-body" title="${escapeHtml(email.body)}">${escapeHtml(email.body)}</td>
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

      // Update badge in the same row
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

function loadFlaggedEmails(userId) {
  emptyState.textContent = "Loading…";
  emptyState.style.display = "";
  emailTable.style.display = "none";

  fetch(`${API_BASE}/flagged?user_id=${encodeURIComponent(userId)}`)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((data) => renderTable(data.emails))
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
