// Wait until page loads
document.addEventListener("DOMContentLoaded", function() {

    // === Chatbot logic ===
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatOutput = document.getElementById("chat-output");

    if (chatForm) {
        chatForm.addEventListener("submit", async function(e) {
            e.preventDefault();
            const question = chatInput.value.trim();
            if (!question) return;

            chatOutput.innerHTML = "<p><em>Thinking...</em></p>";

            const response = await fetch("/ask", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({question: question})
            });

            const data = await response.json();
            chatOutput.innerHTML = `<p>${data.answer}</p>`;
            chatInput.value = "";
        });
    }

    // === Career prediction form ===
    const predictForm = document.getElementById("predict-form");
    const resultDiv = document.getElementById("result");

    if (predictForm) {
        predictForm.addEventListener("submit", async function(e) {
            e.preventDefault();

            const formData = new FormData(predictForm);
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                resultDiv.innerHTML = `<h3>Recommended Major: ${data.major}</h3>
                                       <p>Reason: ${data.reason || "Based on your profile."}</p>`;
            } else {
                resultDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
            }
        });
    }
});
