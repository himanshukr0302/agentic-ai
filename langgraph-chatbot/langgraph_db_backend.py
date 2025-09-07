import os 
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()  
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def chat_node(state: ChatState) -> ChatState:

    # take user input from the state
    messages = state['messages']

    # send to llm
    response = llm.invoke(messages)

    # response store state
    return {'messages': [response]}

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
# checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node',chat_node)

# add edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

# compile
chatbot = graph.compile(checkpointer=checkpointer)

checkpointer.list(None)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) #type: ignore
    
    return list(all_threads)