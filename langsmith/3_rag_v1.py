import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# --- IMPORTS FOR LANGSMITH TRACING ---
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer

# --- IMPORTS FOR RAG APPLICATION ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "islr.pdf"

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index
print("Loading local embedding model...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded. Creating vector store...")
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print("Vector store created.")

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    convert_system_message_to_human=True
)
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# --- LANGSMITH TRACING SETUP ---
# Initialize the LangSmith client and create the tracer for your project
client = Client()
tracer = LangChainTracer(project_name="RAG PDF App", client=client)
# Create a config dictionary to pass the tracer as a callback
config = {"callbacks": [tracer]}

# 6) Ask questions
print("\nPDF RAG ready. Ask a question (or Ctrl+C to exit).")
while True:
    q = input("\nQ: ")
    if q.strip().lower() in ['exit', 'quit']:
        break
    
    # Pass the config to the invoke call to trace the run
    ans = chain.invoke(q.strip(), config=config) #type: ignore
    print("\nA:", ans)