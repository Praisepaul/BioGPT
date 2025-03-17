import torch
from .model_loader import load_model

tokenizer, model = load_model('biogpt')  # Change to 'clinicalbert' if needed

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
