import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load API Key
load_dotenv()

# Configuration
DATA_FOLDER = "data"
DB_PATH = "chroma_db"
BATCH_SIZE = 50

def ingest_new_files():
    # 1. Initialize Database (Do NOT delete it)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 2. Get list of files already in the database
    # We query the DB for all metadata to find unique source filenames
    print("🔍 Checking existing database...")
    existing_data = db.get() # Light metadata fetch
    existing_files = set()
    if existing_data['metadatas']:
        for meta in existing_data['metadatas']:
            if 'source' in meta:
                # Normalize path separators to avoid mismatches
                clean_name = os.path.basename(meta['source'])
                existing_files.add(clean_name)
    
    print(f"   found {len(existing_files)} files already in DB.")

    # 3. Scan 'data' folder for NEW files
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
    new_files = [f for f in all_files if f not in existing_files]

    if not new_files:
        print("✅ No new files to process. Database is up to date!")
        return

    print(f"📦 Found {len(new_files)} new files to ingest: {new_files}")

    # 4. Load ONLY the new files
    raw_documents = []
    for file_name in new_files:
        file_path = os.path.join(DATA_FOLDER, file_name)
        loader = PyPDFLoader(file_path)
        raw_documents.extend(loader.load())

    if not raw_documents:
        print("❌ Error loading documents.")
        return

    # 5. Split Text
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(chunks)} new chunks.")

    # 6. Add to Chroma in Batches
    print("💾 Adding to Database...")
    total_batches = (len(chunks) // BATCH_SIZE) + 1
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_number = (i // BATCH_SIZE) + 1
        
        print(f"   🔹 Processing Batch {batch_number}/{total_batches}...")
        try:
            db.add_documents(batch)
            time.sleep(20) # Respect rate limits
        except Exception as e:
            print(f"      ❌ Error: {e}")
            time.sleep(20)

    print(f"🎉 Successfully added {len(new_files)} files to the database!")

if __name__ == "__main__":
    ingest_new_files()
