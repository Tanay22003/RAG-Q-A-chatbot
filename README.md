# 📊 RAG Q&A Chatbot - Loan Approval Dataset

A chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about loan approval data using document retrieval and LLMs.

## Features
- Converts tabular data into natural text
- Retrieves relevant examples using vector search
- Generates answers with a language model (OpenAI or Mistral)

## Run Locally
```bash
pip install -r requirements.txt
python train_model.py         # optional if using saved embeddings
streamlit run app.py
