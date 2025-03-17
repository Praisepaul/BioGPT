from flask import Flask, request, jsonify
from flask_cors import CORS
from model.inference import generate_response

app = Flask(__name__)
CORS(app)

@app.route('/api/v1/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        response = generate_response(prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
