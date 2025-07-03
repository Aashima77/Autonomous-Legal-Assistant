import os
from chromadb import PersistentClient

if __name__ == "__main__":
    # Path to the ChromaDB embeddings directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    embeddings_dir = os.path.join(repo_root, 'embeddings')

    # Connect to the Chroma persistent client
    client = PersistentClient(path=embeddings_dir)

    # List all collection names
    collection_metadatas = client.list_collections()
    collection_names = [c.name for c in collection_metadatas]
    print("Collections found:", collection_names)

    # For each collection, get the actual object and print its contents
    for name in collection_names:
        collection = client.get_collection(name)
        docs = collection.get(include=["documents", "embeddings"])

        print(f"\n📦 Collection: {name}")
        for i, (doc, embedding) in enumerate(zip(docs['documents'], docs['embeddings'])):
            print(f"📝 Document {i+1}: {doc[:80]}...")
            print(f"🔢 Embedding {i+1}: {embedding[:5]}... [dim={len(embedding)}]\n")
