import os
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load API Key
load_dotenv()

# Configuration
DATA_FOLDER = "data"
DB_PATH = "chroma_db"
BATCH_SIZE = 50  # Process 50 chunks at a time (Safe for Free Tier)

def create_vector_db():
    # 1. Clean up old database if it exists
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("🧹 Cleared old database.")

    print("🔄 Loading PDFs...")
    loader = PyPDFDirectoryLoader(DATA_FOLDER)
    raw_documents = loader.load()
    
    if not raw_documents:
        print("❌ No documents found in 'data' folder.")
        return

    print(f"✅ Loaded {len(raw_documents)} pages.")

    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(chunks)} text chunks.")

    print("💾 Initializing Database...")
    
    # Setup Embedding Model
    # Ensure this matches your app.py exactly!
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Initialize an empty Chroma database
    db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )

    # Process in Batches to avoid Rate Limits
    total_batches = (len(chunks) // BATCH_SIZE) + 1
    
    print(f"🚀 Starting ingestion of {len(chunks)} chunks in {total_batches} batches...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_number = (i // BATCH_SIZE) + 1
        
        print(f"   🔹 Processing Batch {batch_number}/{total_batches} ({len(batch)} chunks)...")
        
        try:
            db.add_documents(batch)
            print("      ✅ Saved.")
        except Exception as e:
            print(f"      ❌ Error in batch {batch_number}: {e}")
            # Optional: Wait longer if error occurs
            time.sleep(60) 

        # ⏳ SLEEP to respect API limits
        # If you are on the free tier, a short pause is required.
        # If you hit 429 errors, increase this to 10 or 20.
        time.sleep(10) 

    print("🎉 Database created successfully!")

if __name__ == "__main__":
    create_vector_db()