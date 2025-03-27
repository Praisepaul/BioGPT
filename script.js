const API_URL = "https://biogpt-1.onrender.com/chat"; // Replace with your actual Render URL

async function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (!userInput) return;

    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;

    document.getElementById("user-input").value = "";

    const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput }),
    });

    const data = await response.json();
    chatBox.innerHTML += `<p><strong>Bot:</strong> ${data.reply}</p>`;
}
