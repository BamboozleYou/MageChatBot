import os
from dotenv import load_dotenv
from google.genai import Client

# Load your API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: API Key not found in .env file")
    exit()

print(f"🔑 Checking models for API Key ending in: ...{api_key[-5:]}")

try:
    client = Client(api_key=api_key)
    # List all models
    models = client.models.list()
    
    print("\n✅ AVAILABLE EMBEDDING MODELS:")
    found_any = False
    for m in models:
        # We are looking for models that support 'embedContent'
        if m.supported_actions and "embedContent" in m.supported_actions:
            print(f" - {m.name}")
            found_any = True
            
    if not found_any:
        print("⚠️ No embedding models found. Your API key might need 'Generative Language API' enabled in Google Cloud Console.")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")