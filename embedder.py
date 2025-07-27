from sentence_transformers import SentenceTransformer
import pandas as pd

def tabular_to_text(row):
    return f"Applicant is a {row['Gender']} {row['Married']} person with {row['Education']} education, income {row['ApplicantIncome']}, loan status {row['Loan_Status']}."

def create_embeddings(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df['text'] = df.apply(tabular_to_text, axis=1)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].tolist())
    
    return df['text'].tolist(), embeddings
