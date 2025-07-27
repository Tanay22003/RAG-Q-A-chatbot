import faiss
from transformers import pipeline

class RAGChatbot:
    def __init__(self, docs, embeddings):
        self.docs = docs
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(embeddings)
        self.generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct", device_map="auto")  # or OpenAI API

    def retrieve(self, query_embedding, top_k=3):
        _, I = self.index.search([query_embedding], top_k)
        return [self.docs[i] for i in I[0]]

    def answer(self, query, embed_model):
        query_embedding = embed_model.encode([query])[0]
        top_docs = self.retrieve(query_embedding)
        context = "\n".join(top_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = self.generator(prompt, max_length=200)[0]['generated_text']
        return response
