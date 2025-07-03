import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import load_and_clean_ipc_text
from utils.chroma import store_embeddings_in_chroma

if __name__ == "__main__":
    # Paths
    texts_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'ipc_law.txt')
    texts_path = os.path.abspath(texts_path)
    embeddings_path = os.path.join(os.path.dirname(__file__), '..', 'embeddings', 'ipc_embeddings.npy')
    embeddings_path = os.path.abspath(embeddings_path)
    chroma_dir = os.path.join(os.path.dirname(__file__), '..', 'embeddings')

    # Load texts and embeddings
    texts = load_and_clean_ipc_text(texts_path)
    embeddings = np.load(embeddings_path, allow_pickle=True)

    # Store in Chroma (batched)
    store_embeddings_in_chroma(texts, embeddings=embeddings, persist_directory=chroma_dir)