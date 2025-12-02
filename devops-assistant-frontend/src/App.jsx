import { useState } from "react";

const API_URL = "http://34.49.135.24/ask-ai";   // ✅ CORRECT ENDPOINT

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    setMessages((prev) => [...prev, { sender: "user", text: input }]);
    const userQuery = input;
    setInput("");
    setLoading(true);

    try {
      console.log('🚀 Sending request to:', `${API_URL}?query=${encodeURIComponent(userQuery)}`);
      
      const res = await fetch(`${API_URL}?query=${encodeURIComponent(userQuery)}`);

      console.log('📥 Response status:', res.status);
      console.log('📥 Response ok:', res.ok);

      if (!res.ok) {
        const errorText = await res.text();
        console.error('❌ Error response:', errorText);
        throw new Error(`HTTP ${res.status}: ${errorText}`);
      }

      const json = await res.json();
      console.log('✅ Parsed JSON:', json);

      const answer = json?.data?.[0]?.answer || "No response from DevOps LLM.";

      setMessages((prev) => [...prev, { sender: "bot", text: answer }]);

    } catch (err) {
      console.error('💥 Full error:', err);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `❌ Error: ${err.message}` }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <h1 className="title">🤖 Ollama DevOps Chat Assistant</h1>

      <div className="chat-box">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.sender}`}>
            {m.text}
          </div>
        ))}

        {loading && <div className="message bot">⏳ Thinking...</div>}
      </div>

      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask me anything about Kubernetes, DevOps, Monitoring..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? "⏳" : "Send"}
        </button>
      </div>
    </div>
  );
}
