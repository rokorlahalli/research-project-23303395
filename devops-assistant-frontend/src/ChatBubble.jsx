import React from "react";

export default function ChatBubble({ sender, text }) {
  return (
    <div className={`bubble ${sender}`}>
      <p>{text}</p>
    </div>
  );
}
