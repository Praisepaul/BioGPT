from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# Load BioGPT or ClinicalBERT
def load_model(model_name):
    if model_name == 'biogpt':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
        model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
    #elif model_name == 'clinicalbert':
    #    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    #    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    else:
        raise ValueError("Invalid model name. Use 'biogpt' or 'clinicalbert'")
    
    return tokenizer, model
