
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='chroma_db')
collection = client.get_or_create_collection('medical')

for f in Path('medical_knowledge').glob('*.txt'):
    txt=f.read_text()
    emb=model.encode(txt).tolist()
    try:
        collection.add(ids=[f.name],documents=[txt],embeddings=[emb])
    except:
        pass

print("Medical knowledge indexed ✅")
