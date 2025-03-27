from transformers import BioGptTokenizer, BioGptForCausalLM

def load_model(model_name='biogpt'):
    if model_name == 'biogpt':
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    else:
        raise ValueError("Invalid model name. Use 'biogpt'")
    return tokenizer, model
