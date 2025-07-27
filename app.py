import streamlit as st
from embedder import create_embeddings
from rag_engine import RAGChatbot
from sentence_transformers import SentenceTransformer

st.title("💬 Loan Dataset Q&A Chatbot")

query = st.text_input("Ask a question about loan approval:")

@st.cache_resource
def load_rag():
    docs, embeddings = create_embeddings("data/Training Dataset.csv")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return RAGChatbot(docs, embeddings), embedder

bot, embedder = load_rag()

if query:
    response = bot.answer(query, embedder)
    st.markdown("### 🤖 Answer")
    st.write(response)
