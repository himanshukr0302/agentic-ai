# pip install -U langchain langchain-google-genai langchain-community faiss-cpu pypdf python-dotenv langsmith sentence-transformers

import os
from dotenv import load_dotenv

from langsmith import traceable, Client
from langchain.callbacks.tracers import LangChainTracer

# Swapped imports from OpenAI to Google and Hugging Face
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_PATH = "islr.pdf" # change to your file

# ---------- traced setup steps ----------
@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    print("Loading local embedding model...")
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(splits, emb)
    print("Vector store created.")
    return vs

# --- CHANGE 1: Added project_name to the main setup decorator ---
@traceable(name="setup_pipeline", project_name="RAG PDF App")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs

# ---------- pipeline ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    convert_system_message_to_human=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Build the index under traced setup
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query (also traced) ----------
print("\nPDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# --- CHANGE 2: Added a specific tracer for the invoke call ---
# This ensures the query trace also goes to the correct project.
client = Client()
tracer = LangChainTracer(project_name="RAG PDF App", client=client)

config = {
    "run_name": "pdf_rag_query",
    "callbacks": [tracer] # Pass the tracer here
}

ans = chain.invoke(q, config=config) #type:ignore
print("\nA:", ans)