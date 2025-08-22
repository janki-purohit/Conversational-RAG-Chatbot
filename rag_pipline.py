from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGPipeline:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def chunk_text(self, text, chunk_size=500):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def build_index(self, chunks):
        self.chunks = chunks
        embeddings = self.embedder.encode(chunks)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(np.array(embeddings))

    def query(self, question, top_k=3):
        q_emb = self.embedder.encode([question])
        distances, indices = self.index.search(np.array(q_emb), top_k)
        return [self.chunks[i] for i in indices[0]]
