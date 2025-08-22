document.addEventListener("DOMContentLoaded", () => {
  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");

  if (!messagesEl || !inputEl || !sendBtn) {
    console.error("UI elements missing");
    return;
  }

  // Always call the backend host you're on (works with /app mount)
  const API_BASE = window.location.origin; // e.g. http://127.0.0.1:8000

  function addMsg(sender, text) {
    const row = document.createElement("div");
    row.className = `message ${sender}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function sendMessage() {
    const text = (inputEl.value || "").trim();
    if (!text) return;

    addMsg("user", text);
    inputEl.value = "";
    inputEl.focus();
    sendBtn.disabled = true;

    // typing indicator
    const typingEl = document.createElement("div");
    typingEl.className = "typing";
    typingEl.textContent = "Assistant is typing…";
    messagesEl.appendChild(typingEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
      const res = await fetch(`${API_BASE}/inquiry`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text })
      });

      const ct = res.headers.get("content-type") || "";
      if (!ct.includes("application/json")) {
        typingEl.remove();
        addMsg("bot", "Unexpected server response. Please refresh and try again.");
        return;
      }

      const data = await res.json();
      typingEl.remove();
      addMsg("bot", data.answer || "Sorry, I couldn't process that.");
    } catch (err) {
      typingEl.remove();
      addMsg("bot", "Network error. Please try again.");
      console.error(err);
    } finally {
      sendBtn.disabled = false;
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });

  // Clear greeting
  addMsg("bot", "Hi! I’m the Patient Inquiry Assistant for the Signifier sleep therapy device. I can help with device usage, setup, appointments and support. I can’t provide medical advice.");
});

