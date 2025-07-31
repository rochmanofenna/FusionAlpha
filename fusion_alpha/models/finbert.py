from transformers import AutoTokenizer, AutoModel
import fin_dataset

def embeddings(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    cls_embed = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return cls_embed

def start_load(model_name="yiyanghkust/finbert-tone"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    return tokenizer, model


if __name__ == "__main__":
    phrasebank_data = fin_dataset.load_phrasebank()
    
    tokenizer, model = start_load("yiyanghkust/finbert-tone")
    
    sample_text = phrasebank_data[0]["sentence"]
    embedding = embeddings(tokenizer, model, sample_text)
    
    print("sample sentence:", sample_text)
    print("CLS embedding shape:", embedding.shape)