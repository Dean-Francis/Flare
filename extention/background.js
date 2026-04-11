const API_BASE = "http://127.0.0.1:8000";

async function callPredict(body) {
	const res = await fetch(`${API_BASE}/predict`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ body }),
	});
	if (!res.ok) throw new Error(`HTTP ${res.status}`);
	return res.json();
}

async function callFlag(user_id, body, label) {
	const res = await fetch(`${API_BASE}/flag`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ user_id, body, label }),
	});
	if (!res.ok) throw new Error(`HTTP ${res.status}`);
	return res.json();
}

// Simple FNV-1a 32-bit hash for content-addressed caching
function hashString(str) {
	let h = 0x811c9dc5;
	for (let i = 0; i < str.length; i++) {
		h ^= str.charCodeAt(i);
		h = (h * 0x01000193) >>> 0;
	}
	return h.toString(16);
}


chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
	if (msg.type === "OPENED_EMAIL") {
		const emailBody = msg.payload.content;
		const cacheKey = `override_${hashString(emailBody)}`;
		chrome.storage.local.get(cacheKey, (items) => {
			const override = items[cacheKey];
			if (override) {
				const result = override;
				chrome.storage.local.set({ lastResult: result });
				chrome.tabs.sendMessage(sender.tab.id, {
					type: "JUDGEMENT",
					judgement: result.predicted,
					predicted: result.predicted,
					confidence: result.confidence,
				});
				chrome.runtime.sendMessage({ type: "SCAN_RESULT", ...result }).catch(() => { });
				return;
			}

			callPredict(emailBody)
				.then((result) => {
					const stored = { ...result, body: emailBody, cacheKey };
					chrome.storage.local.set({ lastResult: stored });
					chrome.tabs.sendMessage(sender.tab.id, {
						type: "JUDGEMENT",
						judgement: result.predicted,
						predicted: result.predicted,
						confidence: result.confidence,
					});
					chrome.runtime.sendMessage({ type: "SCAN_RESULT", ...stored }).catch(() => { });
				})
				.catch((err) => console.error("Predict error:", err));
		});

		return false;
	}


	if (msg.type === "MANUAL_CHECK") {
		const emailBody = msg.body;
		const cacheKey = `override_${hashString(emailBody)}`;

		chrome.storage.local.get(cacheKey, (items) => {
			const override = items[cacheKey];
			if (override) {
				const stored = { ...override, body: emailBody, cacheKey };
				chrome.storage.local.set({ lastResult: stored });
				sendResponse(stored);
				return;
			}

			callPredict(emailBody)
				.then((result) => {
					const stored = { ...result, body: emailBody, cacheKey };
					chrome.storage.local.set({ lastResult: stored });
					sendResponse(stored);
				})
				.catch((err) => {
					console.error("Manual predict error:", err);
					sendResponse(null);
				});
		});

		return true; // keep channel open for async sendResponse
	}


	if (msg.type === "FLAG_EMAIL") {
		chrome.storage.local.get("userId", ({ userId }) => {
			const user_id = userId || "anonymous";
			callFlag(user_id, msg.body, msg.label)
				.then((result) => {
					// Cache the override so rescanning returns the corrected label
					const overrideResult = {
						predicted: msg.label === 1 ? "phishing" : "legitimate",
						confidence: 1.0,
						body: msg.body,
						cacheKey: msg.cacheKey,
					};
					chrome.storage.local.set({
						[msg.cacheKey]: overrideResult,
						lastResult: overrideResult,
					});
					sendResponse({ ok: true, ...result });
				})
				.catch((err) => {
					console.error("Flag error:", err);
					sendResponse({ ok: false });
				});
		});

		return true;
	}
});
