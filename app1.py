import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS  # ✅ Replace Chroma with FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit Title
st.title("RAG Application built on Gemini Model")

# Load PDF (Ensure the PDF is in the same directory)
pdf_path = "BA7204-HUMAN_RESOURCE_MANAGEMENT.pdf"
loader = PyPDFLoader(pdf_path)
data = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# ✅ Use FAISS Instead of Chroma
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)

# Convert VectorStore to a retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Load Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# User Input
query = st.chat_input("Say something: ")

if query:
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you don't know, say so. Keep it concise.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    # Display Response
    st.write(response["answer"])
