import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API Key
load_dotenv()

# Setup Page
st.set_page_config(page_title="Company Helper Bot", page_icon="🤖")
st.title("🤖 Product Assistant")

# 1. Connect to the Database
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5}) # Retrieve top 5 matches

# 2. Setup the Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0) # Temp 0 = strictly factual

# 3. Define the Prompt (The "Rules")
template = """
You are a helpful and professional assistant for Mage Data products.
Answer the user's question based ONLY on the context provided below.

Guidelines:
1. If the context contains a "Video Script" or "Scene" description, use it ONLY if the user explicitly asks about the video, marketing, or visuals.
2. For technical or product questions, prioritize the standard text sections over the video script.
3. If the answer is not in the context, reply exactly: "I don't know, let me connect you with our consultant."
4. Keep your answer concise, professional, and use bullet points if listing features.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. Build the Chain
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. The Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if query := st.chat_input("Ask about our products..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and show answer
    with st.chat_message("assistant"):
        response = rag_chain.invoke(query)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})