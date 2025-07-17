import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize LLM (Groq)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Pinecone settings
index_name = "anypdfbot"  # 🔁 replace with your Pinecone index name

# Load Pinecone vector index
docsearch = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Streamlit App UI
st.set_page_config(page_title="Chat with Your Docs", layout="centered")
st.title("📘 AI Chatbot from Your Book")
st.write("Ask anything from the book you uploaded!")

query = st.text_input("Enter your question:")

if query:
    # Search from vectorstore
    docs = docsearch.similarity_search(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Ask LLM using that context
    prompt = f"Answer the question based only on the following context:\n\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    
    st.markdown("### 🤖 Answer:")
    st.write(response.content)

