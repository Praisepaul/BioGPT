import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load fine-tuned LoRA model
model_name = "Qwen/Qwen2-1.5B-Chat"  # Change if using a local model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Changed to torch.float32 for CPU
    device_map="cpu"  # Change to 'cuda' if GPU is available
)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    inputs = tokenizer(user_input, return_tensors="pt")
    inputs["input_ids"] = inputs["input_ids"].to(torch.long)  # Convert to long tensor

    output = model.generate(**inputs, max_length=1000)

    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"reply": bot_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Get Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)