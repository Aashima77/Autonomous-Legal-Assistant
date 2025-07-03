import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import load_and_clean_ipc_text
from utils.embedder import get_model_and_tokenizer, get_embeddings

if __name__ == "__main__":
    # Path to your raw IPC text file
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'ipc_law.txt')
    file_path = os.path.abspath(file_path)
    
    # Load and clean the text, each line becomes a document
    texts = load_and_clean_ipc_text(file_path)
    
    # Generate embeddings
    tokenizer, model = get_model_and_tokenizer()
    embeddings = get_embeddings(texts, tokenizer, model)

    # Save embeddings to a .npy file
    np.save(os.path.join(os.path.dirname(__file__), '..', 'embeddings', 'ipc_embeddings.npy'), embeddings)
    print(f"✅ Saved {len(embeddings)} embeddings to test_ipc_embeddings.npy")