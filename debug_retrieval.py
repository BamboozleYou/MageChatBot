import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# ⚠️ MAKE SURE THIS MATCHES YOUR INGEST.PY EXACTLY
MODEL_NAME = "models/gemini-embedding-001" 

embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_NAME)
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

query = "What is asset discovery?"
print(f"🔎 Searching for: '{query}'")

results = db.similarity_search(query, k=3)

if not results:
    print("❌ No results found. Database might be empty.")
else:
    print(f"✅ Found {len(results)} chunks:\n")
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:300] + "...") # Show first 300 characters
        print("\n")