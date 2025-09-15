# LangChain & LangSmith Demo: RAG PDF App

This repository contains a series of Python scripts demonstrating various powerful features of the LangChain library, with deep integration for observability using LangSmith. The examples progress from basic chains to complex RAG pipelines, agents with custom tools, and stateful graphs using LangGraph.

The primary project name used for tracing is **"RAG PDF App"**.

## Features Demonstrated

  * **RAG (Retrieval-Augmented Generation)**: Building pipelines to chat with PDF documents.
  * **Local Embeddings**: Using free, open-source Hugging Face models to avoid API rate limits.
  * **Agents**: Creating autonomous agents that can use custom tools (like a weather API) and web search to answer questions.
  * **LangGraph**: Building a multi-step, stateful graph for a complex task (evaluating an essay).
  * **LangSmith Tracing**: Deep integration of LangSmith to trace, monitor, and debug every script.
  * **Caching**: An example of a smart caching system that saves generated vector indices to disk to avoid re-processing files.

## Setup and Installation

These instructions assume you have Python 3.9+ and `uv` installed.

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2\. Create Virtual Environment & Install Dependencies

Use `uv` to create the virtual environment and install all required packages from `requirements.txt`.

```bash
# Create the virtual environment in a .venv folder
uv venv

# Install packages
uv pip install -r requirements.txt
```

### 3\. Set Up Environment Variables

This project requires API keys and configuration settings, which should be stored in a `.env` file. First, create the file by copying the example:

```bash
cp .env.example .env
```

Now, open the `.env` file and add your secret keys. It should look like this:

```env
# Google API Key for Gemini Models
GOOGLE_API_KEY="your_google_api_key_here"

# LangSmith Keys for Tracing
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"

# API Key for the weather tool used in the agent script
WEATHERAPI_API_KEY="your_key_from_weatherapi.com_here"
```

### 4\. Add PDF Document

The RAG scripts expect a PDF file named `islr.pdf` to be in the root directory of the project. Please add your own PDF with this name or update the `PDF_PATH` variable in the scripts.

## Running the Scripts

First, activate the virtual environment:

```bash
source .venv/bin/activate
```

Then, you can run any of the Python scripts. For example:

```bash
# Run the RAG script with caching
python 5_rag_cached_v1.py

# Run the agent script
python 6_agent_v1.py

# Run the LangGraph script
python 7_langgraph_v1.py
```