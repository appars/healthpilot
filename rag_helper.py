
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='chroma_db')
collection = client.get_or_create_collection('medical')

def retrieve_context(query):
    emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[emb], n_results=2)
    docs = results.get('documents',[[]])[0]
    return '\n\n'.join(docs) if docs else 'No medical guidance found.'
