import os
import numpy as np

# Construct the correct path to the .npy file
embeddings_path = os.path.join(os.path.dirname(__file__), '..', 'embeddings', 'ipc_embeddings.npy')
embeddings_path = os.path.abspath(embeddings_path)

# Load the .npy file
embeddings = np.load(embeddings_path, allow_pickle=True)

# View the shape and type
print("Shape:", embeddings.shape)
print("Type:", type(embeddings))

# View the first embedding/vector
print("First embedding:", embeddings[0])

# View the first 3 embeddings (optional)
print("First 3 embeddings:", embeddings[:3])