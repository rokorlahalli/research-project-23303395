// src/services/api.js
const API_URL = import.meta.env.VITE_BACKEND_URL;

export async function askLLM(question) {
  try {
    const res = await fetch(`${API_URL}?query=${encodeURIComponent(question)}`);
    const json = await res.json();
    return json.answer || "No response received.";
  } catch (err) {
    console.error("Error contacting LLM:", err);
    return "❌ Error contacting the DevOps LLM.";
  }
}
