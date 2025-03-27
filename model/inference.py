import torch
from transformers import BioGptForCausalLM, AutoTokenizer
from .model_loader import load_model

# Load BioGPT model and tokenizer
tokenizer, model = load_model('biogpt')

def generate_response(prompt):
    print(f"🟢 Received prompt: {prompt}")  # Debugging log

    # Tokenize input properly
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(model.device)  # Move to correct device

    # Ensure model is in eval mode
    model.eval()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + 100,  # Ensures it generates beyond prompt length
            temperature=0.9,  # Adds diversity to output
            top_p=0.92,  # Nucleus sampling to avoid repetition
            do_sample=True,  # Ensures randomness
            eos_token_id=tokenizer.eos_token_id
        )

    print(f"🔍 Generated output tokens: {output_ids}")  # Debugging log

    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Remove prompt text from generated response
    response_cleaned = response.replace(prompt, "").strip()

    print(f"✅ Decoded response: {response_cleaned}")  # Debugging log

    return response_cleaned
