from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

def get_model_and_tokenizer(model_name="law-ai/InLegalBERT"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model 

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [batch, seq_len, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def get_embeddings(texts, tokenizer, model, batch_size=32):
    embeddings = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            output = model(**encoded_input)
            pooled = mean_pooling(output, encoded_input['attention_mask'])
        embeddings.extend(pooled.cpu().numpy())

    return embeddings


# from transformers import AutoTokenizer, AutoModel
# import torch

# def get_model_and_tokenizer(model_name="law-ai/InLegalBERT"):
#     """
#     Loads the InLegalBERT model and tokenizer.
#     """
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#     model.eval()
#     return tokenizer, model

# def get_embeddings(texts, tokenizer, model):
#     """
#     Generates embeddings using the [CLS] token for each input text.
#     Args:
#         texts (List[str])
#         tokenizer (AutoTokenizer)
#         model (AutoModel)
#     Returns:
#         List[np.ndarray]: List of vector embeddings
#     """
#     embeddings = []
#     with torch.no_grad():
#         for text in texts:
#             encoded_input = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#             output = model(**encoded_input)
#             cls_embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
#             embeddings.append(cls_embedding)
#     return embeddings



# from transformers import AutoTokenizer, AutoModel
# import torch
# from tqdm import tqdm

# def load_inlegalbert(model_name="law-ai/InLegalBERT"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#     model.eval()
#     return tokenizer, model

# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# def get_embeddings(texts, tokenizer, model, batch_size=32):
#     embeddings = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(texts), batch_size)):
#             batch_texts = texts[i:i+batch_size]
#             encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
#             output = model(**encoded_input)
#             pooled_output = mean_pooling(output, encoded_input['attention_mask'])
#             embeddings.extend(pooled_output.cpu().numpy())
#     return embeddings




