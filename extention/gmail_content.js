// content-script.js (add this near the top)

(function injectStyles() {
	if (document.getElementById('my-extension-popup-style')) return;

	const style = document.createElement('style');
	style.id = 'my-extension-popup-style';
	style.textContent = `
    #my-extension-popup {
      position: fixed;
      top: 16px;
      right: 16px;
      width: 340px;
      background: #fff;
      color: #1a1a1a;
      border-radius: 12px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.15), 0 0 0 1px rgba(0,0,0,0.04);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      z-index: 2147483647;
      overflow: hidden;
      border-left: 4px solid #dc2626;
      animation: flare-slide-in 0.25s ease-out;
    }

    @keyframes flare-slide-in {
      from { opacity: 0; transform: translateY(-12px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    #my-extension-popup .flare-top {
      display: flex;
      align-items: flex-start;
      gap: 12px;
      padding: 16px 16px 12px;
    }

    #my-extension-popup .flare-icon {
      flex-shrink: 0;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: #fef2f2;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    #my-extension-popup .flare-icon svg {
      width: 20px;
      height: 20px;
    }

    #my-extension-popup .flare-text {
      flex: 1;
      min-width: 0;
    }

    #my-extension-popup .flare-title {
      font-size: 14px;
      font-weight: 700;
      color: #dc2626;
      margin: 0 0 4px;
      line-height: 1.3;
    }

    #my-extension-popup .flare-desc {
      font-size: 13px;
      color: #4b5563;
      line-height: 1.5;
      margin: 0;
    }

    #my-extension-popup .flare-close {
      position: absolute;
      top: 12px;
      right: 12px;
      border: none;
      background: none;
      color: #9ca3af;
      font-size: 18px;
      cursor: pointer;
      line-height: 1;
      padding: 2px;
      border-radius: 4px;
      transition: background 0.12s, color 0.12s;
    }
    #my-extension-popup .flare-close:hover {
      background: #f3f4f6;
      color: #374151;
    }

    #my-extension-popup .flare-footer {
      padding: 8px 16px;
      background: #fafafa;
      border-top: 1px solid #f3f4f6;
      font-size: 11px;
      color: #9ca3af;
    }
  `;
	document.head.appendChild(style);
})();


// content-script.js

function showExtensionPopup(result) {
	// Avoid duplicates
	if (document.getElementById('my-extension-popup')) return;

	const pct = Math.round(result.confidence * 100);
	const popup = document.createElement('div');
	popup.id = 'my-extension-popup';
	popup.innerHTML = `
    <button class="flare-close">&times;</button>
    <div class="flare-top">
      <div class="flare-icon">
        <svg viewBox="0 0 24 24" fill="none">
          <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"
                stroke="#dc2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="flare-text">
        <p class="flare-title">Phishing Warning</p>
        <p class="flare-desc">This email is likely a phishing attempt (${pct}% confidence). Do not click any links or provide personal information.</p>
      </div>
    </div>
    <div class="flare-footer">Protected by Flare</div>
  `;

	document.body.appendChild(popup);

	popup.querySelector('.flare-close')?.addEventListener('click', () => {
		popup.remove();
	});
}


console.log("Loaded Content Script")
let emailId = null
async function maybeCheckEmail() {
	// Gmail URL changes without reload
	let match = location.hash.match(/#inbox\/([^/]+)/);
	if (!match) {
		match = location.hash.match(/#spam\/([^/]+)/);
	};
	if (!match) return;
	if (emailId == match[1]) return;
	emailId = match[1];
	const listItemEl = document.querySelector("[role=listitem]");
	if (listItemEl === null) return;
	const parsedObj = {
		email: listItemEl.querySelector("[email]").attributes?.email.value,
		name: listItemEl.querySelector("[email]").attributes?.name.value,
		content: listItemEl.innerText,
	}
	console.log(parsedObj)
	chrome.runtime.sendMessage({ type: "OPENED_EMAIL", payload: parsedObj })
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
	console.log(msg);
	if (msg.judgement == "phishing") {
		showExtensionPopup(msg);
	}
})

const observer = new MutationObserver(maybeCheckEmail)
observer.observe(document.body, { childList: true, subtree: true })

