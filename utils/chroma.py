import os
from chromadb import PersistentClient
from utils.embedder import get_model_and_tokenizer, get_embeddings

def store_embeddings_in_chroma(
    texts,
    embeddings=None,
    ids=None,
    metadatas=None,
    persist_directory="embeddings",  # Path to persistent ChromaDB storage
    collection_name="legal_docs",    # Name of the vector collection
    batch_size=1000
):
    # Step 1: Generate document IDs if not provided
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(texts))]

    # Step 2: Generate dummy metadata if not provided
    if metadatas is None:
        metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]

    # Step 3: Generate embeddings if not provided
    if embeddings is None:
        tokenizer, model = get_model_and_tokenizer()
        embeddings = get_embeddings(texts, tokenizer, model)

    # Step 4: Ensure persist directory is absolute
    persist_directory = os.path.abspath(persist_directory)

    # Step 5: Connect to ChromaDB persistent client
    client = PersistentClient(path=persist_directory)

    # Step 6: Get or create the collection
    collection = client.get_or_create_collection(name=collection_name)

    # Step 7: Check for duplicate IDs to avoid overwrite or crashes
    existing_ids = set(collection.get()["ids"])
    duplicate_ids = [doc_id for doc_id in ids if doc_id in existing_ids]
    if duplicate_ids:
        raise ValueError(f"❌ Duplicate IDs detected in collection '{collection_name}': {duplicate_ids[:5]}...")

    # Step 8: Batch insertion
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )

        print(f"✅ Stored batch {i // batch_size + 1} of {len(texts) // batch_size + 1}")

    print(f"\n✅ Successfully stored {len(texts)} embeddings in collection '{collection_name}'")
    print(f"📍 Storage location: {persist_directory}")










"""from langchain_chroma import Chroma
#from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from utils.embedder import get_model_and_tokenizer, get_embeddings

class CustomEmbeddingFunction(Embeddings):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def embed_documents(self, texts):
        # Returns a list of embeddings for a list of texts
        return get_embeddings(texts, self.tokenizer, self.model)

    def embed_query(self, text):
        # Returns a single embedding for a single text
        return get_embeddings([text], self.tokenizer, self.model)[0]

def store_embeddings_in_chroma(
    texts,
    embeddings=None,
    ids=None,
    persist_directory="test_embeddings",
    batch_size=1000
):

    if ids is None:
        ids = [f"doc_{i}" for i in range(len(texts))]
    metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]

    # If embeddings are not provided, generate them
    if embeddings is None:
        tokenizer, model = get_model_and_tokenizer()
        embeddings = get_embeddings(texts, tokenizer, model)
        embedding_function = None  # Will use the embedder
    else:
        # Dummy embedding function class
        class DummyEmbedder:
            def embed_documents(self, texts):
                # Return zeros, won't be used since we pass embeddings directly
                return [[0.0] * len(embeddings[0])] * len(texts)
        embedding_function = DummyEmbedder()

    vectorstore = Chroma(
        collection_name="legal_docs",
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        vectorstore.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids,
            embeddings=batch_embeddings
        )
        print(f"Stored batch {i//batch_size + 1}")

    print(f"Stored {len(texts)} embeddings in ChromaDB at {persist_directory}")"""






"""def store_precomputed_embeddings_in_chroma(texts, embeddings, ids=None, persist_directory="test_embeddings", batch_size=500):
    from langchain_chroma import Chroma

    vectorstore = Chroma(
        collection_name="legal_docs",
        persist_directory=persist_directory
    )

    if ids is None:
        ids = [f"doc_{i}" for i in range(len(texts))]
    metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        vectorstore.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids,
            embeddings=batch_embeddings
        )
        print(f"Stored batch {i//batch_size + 1}")

    print(f"Stored {len(texts)} precomputed embeddings in ChromaDB at {persist_directory}")
"""