import React, { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [text, setText] = useState("I got billed twice and need a refund");
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);

  async function predict() {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    setResult(await res.json());
  }

  async function loadMetrics() {
    const res = await fetch(`${API_BASE}/metrics`);
    setMetrics(await res.json());
  }

  return (
    <div style={{ fontFamily: "sans-serif", maxWidth: 780, margin: "2rem auto" }}>
      <h2>Support Ticket Triage (Semi-supervised)</h2>
      <p>
        Pattern: use unsupervised clustering to pseudo-label unlabeled tickets, then train a
        supervised classifier for production predictions.
      </p>

      <textarea
        rows={5}
        style={{ width: "100%" }}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div style={{ marginTop: 12 }}>
        <button onClick={predict}>Predict Category</button>
        <button onClick={loadMetrics} style={{ marginLeft: 8 }}>
          Show Model Metrics
        </button>
      </div>

      {result && (
        <pre style={{ background: "#f6f8fa", padding: 12, marginTop: 16 }}>
{JSON.stringify(result, null, 2)}
        </pre>
      )}

      {metrics && (
        <pre style={{ background: "#f6f8fa", padding: 12, marginTop: 16 }}>
{JSON.stringify(metrics, null, 2)}
        </pre>
      )}
    </div>
  );
}