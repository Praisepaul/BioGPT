from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load fine-tuned LoRA model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change to your fine-tuned model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"
)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    inputs = tokenizer(user_input, return_tensors="pt").to("cpu")
    output = model.generate(**inputs, max_length=200)

    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"reply": bot_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8610, debug=True)
