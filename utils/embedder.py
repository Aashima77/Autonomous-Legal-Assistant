from transformers import AutoTokenizer, AutoModel
import torch

def get_model_and_tokenizer(model_name="law-ai/InLegalBERT"):
    """
    Loads the InLegalBERT model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def get_embeddings(texts, tokenizer, model):
    """
    Generates embeddings using the [CLS] token for each input text.
    Args:
        texts (List[str])
        tokenizer (AutoTokenizer)
        model (AutoModel)
    Returns:
        List[np.ndarray]: List of vector embeddings
    """
    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded_input = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            output = model(**encoded_input)
            cls_embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
    return embeddings





"""
from transformers import AutoTokenizer, AutoModel
import torch
from loader import load_and_clean_ipc_text

def get_embeddings(texts, model_name="law-ai/InLegalBERT"):
    
    # Given a list of texts, returns their embeddings using InLegalBERT.
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded_input = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            output = model(**encoded_input)
            # Use the [CLS] token representation as the embedding
            cls_embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
    return embeddings

if __name__ == "__main__":
    # Example usage: load cleaned text and get embeddings
    import os
    # Adjust the path as needed
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'test_ipc.txt')
    file_path = os.path.abspath(file_path)
    cleaned_texts = load_and_clean_ipc_text(file_path)
    embeddings = get_embeddings(cleaned_texts)
    print(f"Generated {len(embeddings)} embeddings.")
"""
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
# text = "Replace this string with yours"
# encoded_input = tokenizer(text, return_tensors="pt")
# model = AutoModel.from_pretrained("law-ai/InLegalBERT")
# output = model(**encoded_input)
# last_hidden_state = output.last_hidden_state