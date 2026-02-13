import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever #this line is giving me an error!!!
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Load API Key
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")
```

st.set_page_config(page_title="Document Assistant", page_icon="🤖")
st.title("🤖 Knowledge Base Assistant")

# 1. Connect to Database
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# --- ENSEMBLE RETRIEVER (Semantic + Keyword) ---
# This fixes vague queries and comparisons with ZERO extra API calls.
# Semantic search catches meaning; BM25 catches exact product names/keywords.

# Semantic retriever (your existing approach)
semantic_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# BM25 keyword retriever - load all docs from Chroma once at startup
@st.cache_resource
def build_bm25_retriever():
    """Load all chunks from ChromaDB and build a BM25 keyword index."""
    all_data = db.get(include=["documents", "metadatas"])
    
    if not all_data["documents"]:
        return None
    
    from langchain_core.documents import Document
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 5
    return bm25

bm25_retriever = build_bm25_retriever()

# Combine: 60% semantic, 40% keyword — this balances meaning + exact matches
if bm25_retriever:
    retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
else:
    # Fallback if DB is empty
    retriever = semantic_retriever

# 2. Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2
)

# 3. Universal Prompt — NOT hardcoded to any company or product
template = """
You are a friendly and helpful assistant that answers questions based strictly
on the provided documents.

RULES:
1. Answer ONLY from the context below. Never use outside knowledge or guess.
2. If the context does not contain enough information to answer, reply exactly:
   "I don't know the answer to that. Let me connect you with our consultant."
3. If the user is comparing two or more items, clearly list the key differences
   and similarities using short bullet points for each item.
4. Keep answers concise and conversational — short enough to read comfortably
   in a chat window (aim for 3-6 bullet points max).
5. Adapt your tone to the content: technical for specs, warm for general queries.
6. If the user greets you or makes small talk, respond naturally and warmly,
   then gently guide them to ask a question.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. Chain
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    .assign(answer=(
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    ))
)

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response = chain.invoke(query)
        answer_text = response["answer"]
        source_docs = response["context"]

        st.markdown(answer_text)

        # Dynamic Sources
        if "I don't know" not in answer_text:
            unique_sources = set()
            for doc in source_docs:
                source_name = doc.metadata.get("source", "Unknown")
                clean_name = os.path.basename(source_name)
                page_num = doc.metadata.get("page", 0) + 1
                unique_sources.add(f"{clean_name} (Page {page_num})")

            with st.expander("📚 View Sources"):
                for source in unique_sources:
                    st.caption(f"• {source}")

        st.session_state.messages.append({"role": "assistant", "content": answer_text})
