
import numpy as np

class SemanticRouter():
    def __init__(self, embedding, routes):
        self.routes = routes
        self.embedding = embedding
        self.routesEmbedding = {}

        # Encode samples và lưu trước
        for route in self.routes:
            self.routesEmbedding[route.name] = self.embedding.encode(route.samples)

    def get_routes(self):
        return self.routes

    def guide(self, query):
        # Encode và chuẩn hóa truy vấn
        queryEmbedding = self.embedding.encode([query])
        print("Query_Embedding", queryEmbedding)
        queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding)

        scores = []

        for route in self.routes:
            route_embeds = self.routesEmbedding[route.name]
            # Chuẩn hóa từng vector trong route
            route_embeds = route_embeds / np.linalg.norm(route_embeds, axis=1, keepdims=True)

            # Cosine similarity: dot product vì vector đã chuẩn hóa
            similarities = np.dot(route_embeds, queryEmbedding)
            max_score = np.max(similarities)
            #avg_score = np.mean(similarities)

            scores.append((max_score, route.name))

        scores.sort(reverse=True)
        print("Score,", scores)
        return scores[0]  # (score, route_name)

