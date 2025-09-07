import streamlit as st
from langgraph_backend import chatbot
from  langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# threads
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# for preventing refreshing the chat history
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Enter your text here:")

# running the chatbot
if user_input:

    # first add the message to the history
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # streaming....
    # applying the changes for streaming from here...
    with st.chat_message("assistant"):
    
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream( # type: ignore
            {'messages': [HumanMessage(content=user_input)]},
            config = {'configurable': {'thread_id': 'thread-1'}},
            stream_mode = 'messages'
            ) 
        ) 

    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})