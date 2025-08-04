from langchain_community.embeddings import SentenceTransformerEmbeddings


from typing import List
class EmbeddingsWrapper:
    def __init__(self, base_embedder):
        self.embedder = base_embedder

    def encode(self, texts: List[str]):
        vectors = self.embedder.embed_documents(texts)
        return vectors[0] if len(vectors) == 1 else vectors



