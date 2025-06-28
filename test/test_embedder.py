import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import load_and_clean_ipc_text
from utils.embedder import get_model_and_tokenizer, get_embeddings

# 1. Define the path to your IPC data file
file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'test_ipc.txt')
file_path = os.path.abspath(file_path)

# 2. Load and clean the IPC text
cleaned_texts = load_and_clean_ipc_text(file_path)

# 3. Load model and tokenizer
tokenizer, model = get_model_and_tokenizer()

# 4. Generate embeddings
embeddings = get_embeddings(cleaned_texts, tokenizer, model)

# 5. Print result summary
print(f"✅ Generated {len(embeddings)} embeddings of dimension {embeddings[0].shape[0]}")
